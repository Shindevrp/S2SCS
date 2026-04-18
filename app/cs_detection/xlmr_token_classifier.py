from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from app.utils.logger import get_logger


LABELS = ["AR", "EN", "NE", "AMB", "OTHER"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


@dataclass
class LoadedTokenClassifier:
    tokenizer: object
    model: object


class XLMRCodeSwitchTokenClassifier:
    """XLM-R token classifier for Arabic-English code-switch detection."""

    def __init__(
        self,
        model_name_or_path: str = "xlm-roberta-base",
        device: Optional[str] = None,
        max_length: int = 256,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.logger = get_logger(self.__class__.__name__)

        loaded = self.load_model()
        self.tokenizer = loaded.tokenizer
        self.model = loaded.model

    def load_model(self) -> LoadedTokenClassifier:
        """Load tokenizer + XLM-R token classification model with required label set."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required. Install with `pip install transformers`."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
        )

        model.to(self.device)
        model.eval()

        return LoadedTokenClassifier(tokenizer=tokenizer, model=model)

    def tokenize_text(self, text: str) -> tuple[list[str], object]:
        """Tokenize text to words and subwords with word-id mapping."""
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        return words, encoding

    def align_labels_with_tokens(
        self,
        words: list[str],
        word_labels: list[str],
        encoding: object,
    ) -> list[int]:
        """Align word labels to subword tokens using first-subword labeling."""
        if len(words) != len(word_labels):
            raise ValueError("word_labels length must match number of words")

        label_ids = [LABEL_TO_ID[label] for label in word_labels]
        aligned: list[int] = []
        seen_word_ids: set[int] = set()

        for word_id in encoding.word_ids(batch_index=0):
            if word_id is None:
                aligned.append(-100)
                continue

            if word_id not in seen_word_ids:
                aligned.append(label_ids[word_id])
                seen_word_ids.add(word_id)
            else:
                aligned.append(-100)

        return aligned

    def predict_labels(self, text: str) -> list[tuple[str, str]]:
        """Predict one label per full word and merge subword pieces."""
        if not text or not text.strip():
            return []

        words, encoding = self.tokenize_text(text)
        inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].detach().cpu()

        word_ids = encoding.word_ids(batch_index=0)
        grouped_logits: dict[int, list[torch.Tensor]] = {}

        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            grouped_logits.setdefault(word_id, []).append(logits[token_idx])

        predictions: list[tuple[str, str]] = []
        for word_index, word in enumerate(words):
            if word_index not in grouped_logits:
                continue
            mean_logits = torch.stack(grouped_logits[word_index], dim=0).mean(dim=0)
            label_id = int(torch.argmax(mean_logits).item())
            predictions.append((word, ID_TO_LABEL[label_id]))

        return predictions


_MODEL_SINGLETON: Optional[XLMRCodeSwitchTokenClassifier] = None


def load_model(
    model_name_or_path: str = "xlm-roberta-base",
    device: Optional[str] = None,
    max_length: int = 256,
) -> XLMRCodeSwitchTokenClassifier:
    """Load and cache a model instance for reuse."""
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None:
        _MODEL_SINGLETON = XLMRCodeSwitchTokenClassifier(
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
        )
    return _MODEL_SINGLETON


def predict_labels(text: str) -> list[tuple[str, str]]:
    """Predict labels using the cached model singleton."""
    model = load_model()
    return model.predict_labels(text)


if __name__ == "__main__":
    # Example inference
    sample_text = "انا رايح mall اليوم with Ali"
    model = load_model()
    result = model.predict_labels(sample_text)
    print(result)
