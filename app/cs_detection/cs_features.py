from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from app.dialect.camel_dialect import DEFAULT_CONDITIONING_LABEL, DialectSignal


LABELS = ("AR", "EN", "NE", "OTHER")
LANGUAGE_LABELS = ("AR", "EN")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

DIALECT_LABELS = ("MSA", "Gulf", "Hejazi", "mixed")
DIALECT_TO_ID = {label: index for index, label in enumerate(DIALECT_LABELS)}
ID_TO_DIALECT = {index: label for label, index in DIALECT_TO_ID.items()}


@dataclass
class WordToken:
    text: str
    start_char: int
    end_char: int
    index: int


@dataclass
class TokenPrediction:
    token: str
    label: str
    score: float
    start_char: int
    end_char: int
    dialect_label: str


@dataclass
class CodeSwitchFeatures:
    tokens: list[WordToken]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    dialect_ids: torch.Tensor
    word_ids: list[Optional[int]]


@dataclass
class EmbeddedLanguageIsland:
    language: str
    start_token_index: int
    end_token_index: int
    start_char: int
    end_char: int
    tokens: list[str]
    text: str
    mean_score: float


@dataclass
class CodeSwitchMetrics:
    cs_index: float
    switch_count: int
    valid_transition_count: int
    matrix_language: str
    secondary_language: Optional[str]
    embedded_language_islands: list[EmbeddedLanguageIsland]
    language_token_count: int
    dialect_label: str


def whitespace_tokenize(text: str) -> list[WordToken]:
    tokens: list[WordToken] = []
    start = None

    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                token_text = text[start:index]
                tokens.append(
                    WordToken(
                        text=token_text,
                        start_char=start,
                        end_char=index,
                        index=len(tokens),
                    )
                )
                start = None
            continue

        if start is None:
            start = index

    if start is not None:
        tokens.append(
            WordToken(
                text=text[start:],
                start_char=start,
                end_char=len(text),
                index=len(tokens),
            )
        )

    return tokens


def resolve_dialect_id(dialect_signal: Optional[DialectSignal]) -> int:
    if dialect_signal is None:
        return DIALECT_TO_ID[DEFAULT_CONDITIONING_LABEL]

    label = dialect_signal.conditioning_label or DEFAULT_CONDITIONING_LABEL
    return DIALECT_TO_ID.get(label, DIALECT_TO_ID[DEFAULT_CONDITIONING_LABEL])


def build_code_switch_features(
    text: str,
    tokenizer,
    dialect_signal: DialectSignal,
    max_length: int = 256,
) -> CodeSwitchFeatures:
    if dialect_signal is None:
        raise ValueError(
            "dialect_signal is required because Stage 5 code-switch detection must "
            "consume the Stage 3 dialect signal."
        )

    tokens = whitespace_tokenize(text)
    token_texts = [token.text for token in tokens]

    encoded = tokenizer(
        token_texts,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=max_length,
    )

    word_ids = encoded.word_ids(batch_index=0)
    dialect_id = resolve_dialect_id(dialect_signal)
    dialect_ids = torch.tensor([dialect_id], dtype=torch.long)

    return CodeSwitchFeatures(
        tokens=tokens,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        dialect_ids=dialect_ids,
        word_ids=word_ids,
    )


def compute_cs_index(predictions: list[TokenPrediction]) -> tuple[float, int, int]:
    language_predictions = _language_predictions(predictions)
    if len(language_predictions) < 2:
        return 0.0, 0, 0

    switch_count = sum(
        1
        for previous, current in zip(language_predictions, language_predictions[1:])
        if previous.label != current.label
    )
    valid_transition_count = len(language_predictions) - 1
    cs_index = switch_count / valid_transition_count
    return cs_index, switch_count, valid_transition_count


def detect_matrix_language(predictions: list[TokenPrediction]) -> str:
    language_predictions = _language_predictions(predictions)
    if not language_predictions:
        return "UNKNOWN"

    counts = {label: 0 for label in LANGUAGE_LABELS}
    confidence_sums = {label: 0.0 for label in LANGUAGE_LABELS}

    for prediction in language_predictions:
        counts[prediction.label] += 1
        confidence_sums[prediction.label] += float(prediction.score)

    if counts["AR"] > counts["EN"]:
        return "AR"
    if counts["EN"] > counts["AR"]:
        return "EN"
    if confidence_sums["AR"] > confidence_sums["EN"]:
        return "AR"
    if confidence_sums["EN"] > confidence_sums["AR"]:
        return "EN"

    return language_predictions[0].label


def extract_embedded_language_islands(
    predictions: list[TokenPrediction],
    matrix_language: Optional[str] = None,
) -> list[EmbeddedLanguageIsland]:
    matrix = matrix_language or detect_matrix_language(predictions)
    secondary_language = _secondary_language(matrix)
    if secondary_language is None:
        return []

    islands: list[EmbeddedLanguageIsland] = []
    active_predictions: list[TokenPrediction] = []
    active_start_index: Optional[int] = None

    for token_index, prediction in enumerate(predictions):
        if prediction.label == secondary_language:
            if active_start_index is None:
                active_start_index = token_index
            active_predictions.append(prediction)
            continue

        if active_predictions:
            islands.append(
                _build_island(
                    predictions=active_predictions,
                    language=secondary_language,
                    start_token_index=active_start_index,
                    end_token_index=token_index - 1,
                )
            )
            active_predictions = []
            active_start_index = None

    if active_predictions and active_start_index is not None:
        islands.append(
            _build_island(
                predictions=active_predictions,
                language=secondary_language,
                start_token_index=active_start_index,
                end_token_index=active_start_index + len(active_predictions) - 1,
            )
        )

    return islands


def analyze_code_switch(
    predictions: list[TokenPrediction],
    dialect_label: Optional[str] = None,
) -> CodeSwitchMetrics:
    cs_index, switch_count, valid_transition_count = compute_cs_index(predictions)
    matrix_language = detect_matrix_language(predictions)
    secondary_language = _secondary_language(matrix_language)
    islands = extract_embedded_language_islands(
        predictions=predictions,
        matrix_language=matrix_language,
    )
    language_token_count = len(_language_predictions(predictions))

    resolved_dialect = dialect_label
    if resolved_dialect is None and predictions:
        resolved_dialect = predictions[0].dialect_label
    if resolved_dialect is None:
        resolved_dialect = DEFAULT_CONDITIONING_LABEL

    return CodeSwitchMetrics(
        cs_index=cs_index,
        switch_count=switch_count,
        valid_transition_count=valid_transition_count,
        matrix_language=matrix_language,
        secondary_language=secondary_language,
        embedded_language_islands=islands,
        language_token_count=language_token_count,
        dialect_label=resolved_dialect,
    )


def extract_features(tokens: list[str], labels: list[str]) -> dict[str, object]:
    """Extract simple code-switching features from token-level labels.

    Output format:
    {
      "switch_points": int,
      "CSI": float,
      "matrix_language": str,
      "embedded_islands": list
    }
    """
    if len(tokens) != len(labels):
        raise ValueError("tokens and labels must have the same length")

    total_tokens = len(tokens)
    if total_tokens == 0:
        return {
            "switch_points": 0,
            "CSI": 0.0,
            "matrix_language": "UNKNOWN",
            "embedded_islands": [],
        }

    switch_points = sum(
        1 for previous, current in zip(labels, labels[1:]) if previous != current
    )
    csi = switch_points / float(total_tokens)

    ar_count = sum(1 for label in labels if label == "AR")
    en_count = sum(1 for label in labels if label == "EN")
    if ar_count == 0 and en_count == 0:
        matrix_language = "UNKNOWN"
    elif ar_count >= en_count:
        matrix_language = "AR"
    else:
        matrix_language = "EN"

    embedded_islands: list[dict[str, object]] = []
    if matrix_language in {"AR", "EN"}:
        non_dominant = "EN" if matrix_language == "AR" else "AR"
        island_start: Optional[int] = None

        for index, label in enumerate(labels):
            if label == non_dominant:
                if island_start is None:
                    island_start = index
                continue

            if island_start is not None:
                island_end = index - 1
                embedded_islands.append(
                    {
                        "language": non_dominant,
                        "start": island_start,
                        "end": island_end,
                        "tokens": tokens[island_start : island_end + 1],
                        "text": " ".join(tokens[island_start : island_end + 1]),
                    }
                )
                island_start = None

        if island_start is not None:
            island_end = total_tokens - 1
            embedded_islands.append(
                {
                    "language": non_dominant,
                    "start": island_start,
                    "end": island_end,
                    "tokens": tokens[island_start : island_end + 1],
                    "text": " ".join(tokens[island_start : island_end + 1]),
                }
            )

    return {
        "switch_points": switch_points,
        "CSI": csi,
        "matrix_language": matrix_language,
        "embedded_islands": embedded_islands,
    }


def _language_predictions(predictions: list[TokenPrediction]) -> list[TokenPrediction]:
    return [prediction for prediction in predictions if prediction.label in LANGUAGE_LABELS]


def _secondary_language(matrix_language: str) -> Optional[str]:
    if matrix_language == "AR":
        return "EN"
    if matrix_language == "EN":
        return "AR"
    return None


def _build_island(
    predictions: list[TokenPrediction],
    language: str,
    start_token_index: int,
    end_token_index: int,
) -> EmbeddedLanguageIsland:
    mean_score = sum(prediction.score for prediction in predictions) / len(predictions)
    token_texts = [prediction.token for prediction in predictions]

    return EmbeddedLanguageIsland(
        language=language,
        start_token_index=start_token_index,
        end_token_index=end_token_index,
        start_char=predictions[0].start_char,
        end_char=predictions[-1].end_char,
        tokens=token_texts,
        text=" ".join(token_texts),
        mean_score=mean_score,
    )
