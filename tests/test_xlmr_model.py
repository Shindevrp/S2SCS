import torch
from torch import nn

from app.cs_detection.xlmr_model import (
    CodeSwitchResult,
    DialectAwareXLMRTokenClassifier,
    XLMRCodeSwitchDetector,
)
from app.dialect.camel_dialect import DialectSignal


class FakeBackboneOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape
        hidden_size = self.config.hidden_size
        hidden = torch.arange(
            batch_size * sequence_length * hidden_size,
            dtype=torch.float32,
        ).reshape(batch_size, sequence_length, hidden_size)
        return FakeBackboneOutput(last_hidden_state=hidden)


class FakeBatchEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeTokenizer:
    def __call__(
        self,
        token_texts,
        is_split_into_words,
        return_tensors,
        truncation,
        padding,
        max_length,
    ):
        return FakeBatchEncoding(
            input_ids=torch.tensor([[0, 21, 22, 31, 2]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
            word_ids=[None, 0, 0, 1, None],
        )


class FakeInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_dialect_ids = None

    def forward(self, input_ids, attention_mask, dialect_ids):
        self.last_dialect_ids = dialect_ids.clone()
        return torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [5.0, 1.0, 0.0, 0.0],
                    [4.0, 1.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.9,
        raw_label="MSA" if label == "MSA" else "RIY",
        raw_city=None,
        raw_country=None,
        raw_region=None,
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.8, "Hejazi": 0.1},
        is_fallback=False,
    )


def test_dialect_aware_xlmr_classifier_forward_shape() -> None:
    classifier = DialectAwareXLMRTokenClassifier(
        backbone=FakeBackbone(hidden_size=8),
        hidden_size=8,
        dialect_embedding_dim=4,
    )

    logits = classifier(
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        dialect_ids=torch.tensor([1]),
    )

    assert logits.shape == (1, 4, 4)


def test_predict_tokens_aggregates_subwords_to_word_labels() -> None:
    detector = XLMRCodeSwitchDetector(
        tokenizer=FakeTokenizer(),
        model=FakeInferenceModel(),
        device="cpu",
    )

    result = detector.predict_tokens("hello مرحبا", dialect_signal=build_signal("Gulf"))

    assert isinstance(result, CodeSwitchResult)
    assert [prediction.token for prediction in result.predictions] == ["hello", "مرحبا"]
    assert [prediction.label for prediction in result.predictions] == ["AR", "EN"]
    assert all(prediction.dialect_label == "Gulf" for prediction in result.predictions)
    assert int(detector.model.last_dialect_ids[0].item()) == 1


def test_predict_tokens_rejects_empty_text() -> None:
    detector = XLMRCodeSwitchDetector(
        tokenizer=FakeTokenizer(),
        model=FakeInferenceModel(),
        device="cpu",
    )

    try:
        detector.predict_tokens("   ", dialect_signal=build_signal("Gulf"))
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty text")


def test_predict_tokens_requires_dialect_signal() -> None:
    detector = XLMRCodeSwitchDetector(
        tokenizer=FakeTokenizer(),
        model=FakeInferenceModel(),
        device="cpu",
    )

    try:
        detector.predict_tokens("hello مرحبا", dialect_signal=None)
    except ValueError as exc:
        assert "dialect_signal is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when dialect_signal is missing")
