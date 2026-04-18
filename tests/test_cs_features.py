import torch

from app.cs_detection.cs_features import build_code_switch_features, whitespace_tokenize
from app.dialect.camel_dialect import DialectSignal


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
        assert is_split_into_words is True
        assert return_tensors == "pt"
        return FakeBatchEncoding(
            input_ids=torch.tensor([[0, 11, 12, 13, 2]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
            word_ids=[None, 0, 0, 1, None],
        )


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.9,
        raw_label="MSA" if label == "MSA" else "JED",
        raw_city=None,
        raw_country=None,
        raw_region=None,
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.1, "Hejazi": 0.8},
        is_fallback=False,
    )


def test_whitespace_tokenize_preserves_offsets() -> None:
    tokens = whitespace_tokenize("hello   مرحبا world")

    assert [token.text for token in tokens] == ["hello", "مرحبا", "world"]
    assert tokens[0].start_char == 0
    assert tokens[1].start_char == 8
    assert tokens[1].end_char == 13


def test_build_code_switch_features_uses_dialect_signal() -> None:
    features = build_code_switch_features(
        text="hello مرحبا",
        tokenizer=FakeTokenizer(),
        dialect_signal=build_signal("Hejazi"),
    )

    assert [token.text for token in features.tokens] == ["hello", "مرحبا"]
    assert features.word_ids == [None, 0, 0, 1, None]
    assert int(features.dialect_ids[0].item()) == 2
