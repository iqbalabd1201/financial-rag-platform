"""Fine-tune bge-reranker-base with PEFT/LoRA on domain-specific pairs.

This is the SAME base model that failed off-the-shelf (36.6% hit@5 after
reranking, vs 58.3% without -- see src/retrieval/reranker.py's docstring
and docs/failure_analysis.md). The goal isn't to replace it with a bigger
model; it's to teach this specific model the domain's relevance patterns
(tables of raw figures, not narrative prose) via LoRA, using the hard
negatives built by build_reranker_training_data.py.

Saves ONLY the LoRA adapter (a few MB), not a merged model (~1GB+) -- the
adapter is loaded on top of the base model at inference time. See
src/retrieval/reranker.py's load_finetuned_reranker() for how it's used.

Needs a GPU. Run in Colab:
    !pip install -q peft transformers accelerate
    !python scripts/finetune_reranker_lora.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "BAAI/bge-reranker-base"
TRAIN_PAIRS_PATH = "data/qa_gold/reranker_train_pairs.jsonl"
ADAPTER_OUT_DIR = "models/reranker_lora_adapter"
MAX_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        enc = self.tokenizer(
            p["question"], p["page_text"],
            truncation=True, max_length=MAX_LENGTH, padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(p["label"], dtype=torch.float),
        }


def load_pairs(path: str):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected -- this will be very slow. "
              "In Colab: Runtime > Change runtime type > GPU.")

    print(f"\n=== Step 1: load training pairs from {TRAIN_PAIRS_PATH} ===")
    pairs = load_pairs(TRAIN_PAIRS_PATH)
    print(f"Loaded {len(pairs)} pairs.")

    print(f"\n=== Step 2: load base model + tokenizer ({BASE_MODEL}) ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)

    print("\n=== Step 3: apply LoRA adapter ===")
    # target_modules=["query", "value"] matches the attention projection
    # naming used by bge-reranker-base's underlying (RoBERTa-family)
    # architecture -- the standard LoRA target for BERT/RoBERTa-style models.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n=== Step 4: train ===")
    dataset = PairDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        print(f"  Epoch {epoch + 1}/{EPOCHS}: avg loss = {epoch_loss / len(loader):.4f}")

    print(f"\n=== Step 5: save LoRA adapter to {ADAPTER_OUT_DIR} ===")
    os.makedirs(ADAPTER_OUT_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUT_DIR)
    print("Done. Adapter size:")
    os.system(f"du -sh {ADAPTER_OUT_DIR}")
    print("\nNext: copy this folder to the repo (or Drive), then run "
          "scripts/evaluate_finetuned_reranker.py on the 15 held-out questions.")


if __name__ == "__main__":
    main()