import torch

from app.cs_detection.xlmr_token_classifier import (
    LABEL_TO_ID,
    XLMRCodeSwitchTokenClassifier,
)


class FakeEncoding(dict):
    def __init__(self, word_ids):
        super().__init__({"input_ids": torch.tensor([[0, 1, 2, 3, 4, 5]])})
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class FakeOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class FakeModel:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    def __call__(self, **kwargs):
        return FakeOutput(self._logits)


def test_align_labels_with_tokens_first_subword_only() -> None:
    classifier = XLMRCodeSwitchTokenClassifier.__new__(XLMRCodeSwitchTokenClassifier)
    words = ["انا", "hello"]
    labels = ["AR", "EN"]
    encoding = FakeEncoding([None, 0, 0, 1, None])

    aligned = classifier.align_labels_with_tokens(words, labels, encoding)

    assert aligned == [-100, LABEL_TO_ID["AR"], -100, LABEL_TO_ID["EN"], -100]


def test_predict_labels_merges_subword_logits_to_word_label() -> None:
    # token ids map: [None, word0, word0, word1, word1, None]
    encoding = FakeEncoding([None, 0, 0, 1, 1, None])
    words = ["انا", "hello"]

    # labels: [AR, EN, NE, AMB, OTHER]
    # word0 avg logits highest at AR; word1 avg logits highest at EN
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 1.0, 0.1, 0.0, 0.0],
                [2.0, 1.0, 0.1, 0.0, 0.0],
                [0.5, 2.5, 0.2, 0.0, 0.0],
                [0.4, 2.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )

    classifier = XLMRCodeSwitchTokenClassifier.__new__(XLMRCodeSwitchTokenClassifier)
    classifier.device = "cpu"
    classifier.model = FakeModel(logits)

    def _fake_tokenize_text(text: str):
        return words, encoding

    classifier.tokenize_text = _fake_tokenize_text

    result = classifier.predict_labels("dummy text")

    assert result == [("انا", "AR"), ("hello", "EN")]
