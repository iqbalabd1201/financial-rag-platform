"""Cross-encoder reranking -- IMPLEMENTED, TESTED, AND DISABLED BY DEFAULT."""
from sentence_transformers import CrossEncoder


def load_reranker(model_name: str = "BAAI/bge-reranker-base", device: str = "cuda"):
    return CrossEncoder(model_name, max_length=512, device=device)


class FinetunedRerankerWrapper:
    def __init__(self, adapter_path: str, base_model_name: str = "BAAI/bge-reranker-base",
                 device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, pairs, batch_size: int = 16, show_progress_bar: bool = False):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            queries = [p[0] for p in batch]
            texts = [p[1] for p in batch]
            enc = self.tokenizer(
                queries, texts, truncation=True, max_length=512,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())
        return scores


def load_finetuned_reranker(adapter_path: str = "models/reranker_lora_adapter",
                              base_model_name: str = "BAAI/bge-reranker-base",
                              device: str = "cpu") -> FinetunedRerankerWrapper:
    return FinetunedRerankerWrapper(adapter_path, base_model_name, device)


def rerank(reranker, query: str, candidates: list, page_text_lookup: dict) -> list:
    pairs = [[query, page_text_lookup[(doc_id, pn)]] for doc_id, pn in candidates]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    return [c for c, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
