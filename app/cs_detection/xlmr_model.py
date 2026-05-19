from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from app.cs_detection.cs_features import (
    DIALECT_LABELS,
    ID_TO_LABEL,
    LABELS,
    CodeSwitchFeatures,
    TokenPrediction,
    build_code_switch_features,
)
from app.dialect.camel_dialect import DialectSignal
from app.utils.logger import get_logger


DEFAULT_XLMR_MODEL = "models/1716Shinde/xlmr-cs-finetuned"


@dataclass
class CodeSwitchResult:
    text: str
    dialect_label: str
    predictions: list[TokenPrediction]


class DialectAwareXLMRTokenClassifier(nn.Module):
    """XLM-R token classifier that injects one dialect embedding per sequence."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_labels: int = len(LABELS),
        num_dialects: int = len(DIALECT_LABELS),
        dialect_embedding_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.dialect_embedding = nn.Embedding(num_dialects, dialect_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + dialect_embedding_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dialect_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        dialect_features = self.dialect_embedding(dialect_ids)
        dialect_features = dialect_features.unsqueeze(1).expand(
            -1,
            sequence_output.size(1),
            -1,
        )

        combined = torch.cat([sequence_output, dialect_features], dim=-1)
        combined = self.dropout(combined)
        return self.classifier(combined)


class XLMRCodeSwitchDetector:
    """Inference wrapper for token-level code-switch detection conditioned on Stage 3 dialect."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_XLMR_MODEL,
        device: Optional[str] = None,
        max_length: int = 256,
        tokenizer: Optional[Any] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.logger = get_logger(self.__class__.__name__)

        self.tokenizer = tokenizer
        self.model = model
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model = self._load_components()

        self.model.to(self.device)
        self.model.eval()

    def predict_tokens(
        self,
        text: str,
        dialect_signal: DialectSignal,
    ) -> CodeSwitchResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        if dialect_signal is None:
            raise ValueError(
                "dialect_signal is required because Stage 5 must use Stage 3 dialect output "
                "as an explicit input feature."
            )

        features = build_code_switch_features(
            text=text,
            tokenizer=self.tokenizer,
            dialect_signal=dialect_signal,
            max_length=self.max_length,
        )

        if not features.tokens:
            return CodeSwitchResult(
                text=text,
                dialect_label=dialect_signal.conditioning_label,
                predictions=[],
            )

        inputs = self._move_features_to_device(features)

        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    dialect_ids=inputs.dialect_ids,
                )
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
        except Exception as exc:
            self.logger.exception("Code-switch detection failed")
            raise RuntimeError("XLM-R code-switch detection failed") from exc

        predictions = self._aggregate_word_predictions(features, logits.cpu())
        dialect_label = dialect_signal.conditioning_label

        self.logger.debug(
            "dialect_label=%s tokens=%s predictions=%s",
            dialect_label,
            len(features.tokens),
            len(predictions),
        )

        return CodeSwitchResult(
            text=text,
            dialect_label=dialect_label,
            predictions=predictions,
        )

    def _load_components(self) -> tuple[Any, nn.Module]:
        try:
            from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel
        except ImportError as exc:
            self.logger.exception("transformers is not installed")
            raise RuntimeError(
                "transformers is required to load the XLM-R code-switch detector."
            ) from exc

        try:
            config_path = Path(self.model_name_or_path)
            from transformers import XLMRobertaTokenizerFast
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(config_path)
            # Load config directly to avoid auto_map issues with custom model_type
            config = XLMRobertaConfig.from_pretrained(config_path)

            config = XLMRobertaConfig.from_pretrained(config_path)
            backbone = XLMRobertaModel(config)
            model = DialectAwareXLMRTokenClassifier(
                backbone=backbone,
                hidden_size=config.hidden_size,
            )

            state_dict = torch.load(
                config_path / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )

            remapped = {}
            for key, value in state_dict.items():
                if key.startswith("xlmr."):
                    remapped[f"backbone.{key[len('xlmr.'):]}"] = value
                elif key == "dialect_embeddings.weight":
                    remapped["dialect_embedding.weight"] = value
                elif key.startswith("classifier."):
                    remapped[key] = value
                else:
                    remapped[key] = value

            model.load_state_dict(remapped, strict=True)
            return tokenizer, model
        except Exception as exc:
            self.logger.exception("Failed to load code-switch model from %s", self.model_name_or_path)
            raise RuntimeError(
                f"Failed to load code-switch model from {self.model_name_or_path}"
            ) from exc

    def _move_features_to_device(self, features: CodeSwitchFeatures) -> CodeSwitchFeatures:
        return CodeSwitchFeatures(
            tokens=features.tokens,
            input_ids=features.input_ids.to(self.device),
            attention_mask=features.attention_mask.to(self.device),
            dialect_ids=features.dialect_ids.to(self.device),
            word_ids=features.word_ids,
        )

    def _aggregate_word_predictions(
        self,
        features: CodeSwitchFeatures,
        logits: torch.Tensor,
    ) -> list[TokenPrediction]:
        token_logits = logits[0]
        grouped_logits: dict[int, list[torch.Tensor]] = {}

        for token_index, word_id in enumerate(features.word_ids):
            if word_id is None:
                continue
            grouped_logits.setdefault(word_id, []).append(token_logits[token_index])

        predictions: list[TokenPrediction] = []
        for word in features.tokens:
            group = grouped_logits.get(word.index)
            if not group:
                continue

            mean_logits = torch.stack(group, dim=0).mean(dim=0)
            probabilities = torch.softmax(mean_logits, dim=-1)
            label_id = int(torch.argmax(probabilities).item())
            score = float(probabilities[label_id].item())

            predictions.append(
                TokenPrediction(
                    token=word.text,
                    label=ID_TO_LABEL[label_id],
                    score=score,
                    start_char=word.start_char,
                    end_char=word.end_char,
                    dialect_label="",
                )
            )

        dialect_id = int(features.dialect_ids[0].item())
        dialect_label = DIALECT_LABELS[dialect_id]
        for prediction in predictions:
            prediction.dialect_label = dialect_label

        return predictions
