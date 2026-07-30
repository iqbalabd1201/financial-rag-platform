import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "BAAI/bge-reranker-base"
MAX_LENGTH = 512
BATCH_SIZE = 8
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


def load_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--train_pairs", default="data/qa_gold/reranker_train_pairs.jsonl")
    parser.add_argument("--out_dir", default="models/reranker_lora_adapter")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.fold is not None:
        train_pairs_path = f"data/qa_gold/reranker_fold{args.fold}_train_pairs.jsonl"
        out_dir = f"models/reranker_lora_adapter_fold{args.fold}"
    else:
        train_pairs_path = args.train_pairs
        out_dir = args.out_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\n=== Step 1: load training pairs from {train_pairs_path} ===")
    pairs = load_pairs(train_pairs_path)
    print(f"Loaded {len(pairs)} pairs.")

    print(f"\n=== Step 2: load base model + tokenizer ({BASE_MODEL}) ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.to(device)

    print("\n=== Step 3: apply LoRA adapter ===")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n=== Step 4: train ({args.epochs} epochs) ===")
    dataset = PairDataset(pairs, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
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

        print(f"  Epoch {epoch + 1}/{args.epochs}: avg loss = {epoch_loss / len(loader):.4f}")

    print(f"\n=== Step 5: save LoRA adapter to {out_dir} ===")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done.")
    os.system(f"du -sh {out_dir}")


if __name__ == "__main__":
    main()
