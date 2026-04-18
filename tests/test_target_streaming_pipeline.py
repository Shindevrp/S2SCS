from dataclasses import dataclass

from app.cs_detection.cs_features import TokenPrediction, whitespace_tokenize
from app.cs_detection.xlmr_model import CodeSwitchResult
from app.dialect.camel_dialect import DialectSignal
from app.normalization.arabic_normalizer import NormalizationResult
from app.pipeline.target_streaming_pipeline import StreamingReasoningOrchestrator


def build_signal() -> DialectSignal:
    return DialectSignal(
        conditioning_label="Gulf",
        confidence=0.81,
        raw_label="RIY",
        raw_city="Riyadh",
        raw_country="Saudi Arabia",
        raw_region="Gulf",
        normalized_text="مرحبا",
        bucket_scores={"MSA": 0.1, "Gulf": 0.8, "Hejazi": 0.1},
        is_fallback=False,
    )


class FakeDialectIdentifier:
    def __init__(self) -> None:
        self.signal = build_signal()
        self.last_text = None

    def identify(self, text: str) -> DialectSignal:
        self.last_text = text
        return self.signal


class FakeNormalizer:
    def __init__(self) -> None:
        self.last_text = None

    def normalize(self, text: str, dialect_signal: DialectSignal) -> NormalizationResult:
        self.last_text = text
        normalized_text = text.lower().strip()
        return NormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            dialect_label=dialect_signal.conditioning_label,
            applied_rules=["lowercase_for_test"],
        )


class FakeCodeSwitchDetector:
    def __init__(self) -> None:
        self.last_text = None

    def predict_tokens(self, text: str, dialect_signal: DialectSignal) -> CodeSwitchResult:
        self.last_text = text
        predictions: list[TokenPrediction] = []
        for index, token in enumerate(whitespace_tokenize(text)):
            label = "AR" if index % 2 == 0 else "EN"
            predictions.append(
                TokenPrediction(
                    token=token.text,
                    label=label,
                    score=0.9,
                    start_char=token.start_char,
                    end_char=token.end_char,
                    dialect_label=dialect_signal.conditioning_label,
                )
            )

        return CodeSwitchResult(
            text=text,
            dialect_label=dialect_signal.conditioning_label,
            predictions=predictions,
        )


@dataclass
class FakeResponse:
    response_text: str


class FakeResponseGenerator:
    def __init__(self) -> None:
        self.last_prompt_input = None

    def generate_response(self, prompt_input):
        self.last_prompt_input = prompt_input
        return FakeResponse(response_text="streamed reply")


def build_orchestrator() -> StreamingReasoningOrchestrator:
    return StreamingReasoningOrchestrator(
        dialect_identifier=FakeDialectIdentifier(),
        text_normalizer=FakeNormalizer(),
        code_switch_detector=FakeCodeSwitchDetector(),
        response_generator=FakeResponseGenerator(),
        sliding_window_tokens=4,
        max_context_updates=3,
    )


def test_process_partial_transcript_uses_sliding_window_for_detector() -> None:
    orchestrator = build_orchestrator()

    update = orchestrator.process_partial_transcript(
        "one two three four five six seven"
    )

    assert update.sliding_window_text == "four five six seven"
    assert update.code_switch_result.text == "four five six seven"


def test_generate_turn_response_uses_context_buffer_suffix() -> None:
    orchestrator = build_orchestrator()

    orchestrator.process_partial_transcript("alpha beta gamma")
    orchestrator.process_partial_transcript("alpha beta gamma delta")
    turn_response = orchestrator.generate_turn_response(
        task_instruction="Respond briefly."
    )

    task_instruction = turn_response.prompt_input.task_instruction
    assert "Respond briefly." in task_instruction
    assert "Real-time context buffer:" in task_instruction
    assert "Latest matrix language" in task_instruction
    assert turn_response.llm_response.response_text == "streamed reply"


def test_build_turn_prompt_input_includes_rolling_context_suffix() -> None:
    orchestrator = build_orchestrator()

    orchestrator.process_partial_transcript("one two")
    prompt_input = orchestrator.build_turn_prompt_input(task_instruction="Be concise.")

    assert "Be concise." in prompt_input.task_instruction
    assert "Real-time context buffer:" in prompt_input.task_instruction
    assert "Average rolling CSI" in prompt_input.task_instruction


def test_reset_turn_clears_update_state() -> None:
    orchestrator = build_orchestrator()
    orchestrator.process_partial_transcript("alpha beta")

    orchestrator.reset_turn()

    assert orchestrator.context_buffer.user_updates == []
