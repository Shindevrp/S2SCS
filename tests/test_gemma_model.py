import torch

from app.cs_detection.cs_features import CodeSwitchMetrics
from app.dialect.camel_dialect import DialectSignal
from app.llm.gemma_model import GemmaResponseGenerator
from app.llm.prompt_builder import ResponsePromptInput


class FakeBatchEncoding(dict):
    def to(self, device):
        moved = FakeBatchEncoding()
        for key, value in self.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 99
        self.last_prompt = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.last_prompt = messages[0]["content"]
        return f"<chat>{self.last_prompt}</chat>"

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor([[10, 11, 12]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens is True
        assert token_ids.tolist() == [42, 43]
        return "أكيد، let's do it"


class FakeModel:
    def __init__(self) -> None:
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return torch.tensor([[10, 11, 12, 42, 43]])


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.91,
        raw_label="RIY" if label == "Gulf" else "MSA",
        raw_city=None,
        raw_country=None,
        raw_region=None,
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.8, "Hejazi": 0.1},
        is_fallback=False,
    )


def build_prompt_input() -> ResponsePromptInput:
    metrics = CodeSwitchMetrics(
        cs_index=0.33,
        switch_count=1,
        valid_transition_count=3,
        matrix_language="AR",
        secondary_language="EN",
        embedded_language_islands=[],
        language_token_count=4,
        dialect_label="Gulf",
    )
    return ResponsePromptInput(
        normalized_text="انا ready",
        dialect_signal=build_signal("Gulf"),
        code_switch_metrics=metrics,
    )


def test_generate_response_returns_generated_text_only() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    generator = GemmaResponseGenerator(
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    result = generator.generate_response(build_prompt_input())

    assert result.response_text == "أكيد، let's do it"
    assert result.model_name_or_path == "google/gemma-2b-it"
    assert "انا ready" in result.prompt
    assert tokenizer.last_prompt is not None
    assert model.last_kwargs["max_new_tokens"] == 128


def test_generate_response_uses_plain_prompt_when_chat_template_is_missing() -> None:
    class PlainTokenizer(FakeTokenizer):
        apply_chat_template = None

    tokenizer = PlainTokenizer()
    model = FakeModel()
    generator = GemmaResponseGenerator(
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    result = generator.generate_response(build_prompt_input())

    assert result.response_text == "أكيد، let's do it"
