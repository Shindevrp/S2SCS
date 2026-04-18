from app.cs_detection.cs_features import TokenPrediction
from app.cs_detection.xlmr_model import CodeSwitchResult
from app.dialect.camel_dialect import DialectSignal
from app.llm.gemma_model import GemmaResponse
from app.normalization.arabic_normalizer import NormalizationResult
from app.pipeline.main_pipeline import Stage3ConditionedTextPipeline


def build_signal() -> DialectSignal:
    return DialectSignal(
        conditioning_label="Hejazi",
        confidence=0.88,
        raw_label="JED",
        raw_city="Jeddah",
        raw_country="Saudi Arabia",
        raw_region="Gulf",
        normalized_text="يا هلا",
        bucket_scores={"MSA": 0.05, "Gulf": 0.10, "Hejazi": 0.85},
        is_fallback=False,
    )


class FakeDialectIdentifier:
    def __init__(self, dialect_signal: DialectSignal) -> None:
        self.dialect_signal = dialect_signal
        self.last_text = None

    def identify(self, text: str) -> DialectSignal:
        self.last_text = text
        return self.dialect_signal


class FakeNormalizer:
    def __init__(self) -> None:
        self.last_text = None
        self.last_dialect_signal = None

    def normalize(self, text: str, dialect_signal: DialectSignal) -> NormalizationResult:
        self.last_text = text
        self.last_dialect_signal = dialect_signal
        return NormalizationResult(
            original_text=text,
            normalized_text="يا هلا hello",
            dialect_label=dialect_signal.conditioning_label,
            applied_rules=["dialect_arabizi_map:hejazi"],
        )


class FakeCodeSwitchDetector:
    def __init__(self) -> None:
        self.last_text = None
        self.last_dialect_signal = None

    def predict_tokens(self, text: str, dialect_signal: DialectSignal) -> CodeSwitchResult:
        self.last_text = text
        self.last_dialect_signal = dialect_signal
        return CodeSwitchResult(
            text=text,
            dialect_label=dialect_signal.conditioning_label,
            predictions=[
                TokenPrediction("يا", "AR", 0.95, 0, 2, dialect_signal.conditioning_label),
                TokenPrediction("هلا", "AR", 0.94, 3, 6, dialect_signal.conditioning_label),
                TokenPrediction("hello", "EN", 0.97, 7, 12, dialect_signal.conditioning_label),
            ],
        )


class FakeResponseGenerator:
    def __init__(self) -> None:
        self.last_prompt_input = None

    def generate_response(self, prompt_input):
        self.last_prompt_input = prompt_input
        return GemmaResponse(
            prompt="prompt",
            response_text="يا هلا hello back",
            model_name_or_path="google/gemma-2b-it",
        )


def test_stage3_dialect_signal_explicitly_conditions_stages_4_5_and_7() -> None:
    dialect_signal = build_signal()
    dialect_identifier = FakeDialectIdentifier(dialect_signal)
    normalizer = FakeNormalizer()
    code_switch_detector = FakeCodeSwitchDetector()
    response_generator = FakeResponseGenerator()

    pipeline = Stage3ConditionedTextPipeline(
        dialect_identifier=dialect_identifier,
        text_normalizer=normalizer,
        code_switch_detector=code_switch_detector,
        response_generator=response_generator,
    )

    result = pipeline.generate_response(
        "ياهلا hello",
        task_instruction="Reply in a friendly code-switched style.",
    )

    assert dialect_identifier.last_text == "ياهلا hello"
    assert normalizer.last_dialect_signal is dialect_signal
    assert code_switch_detector.last_dialect_signal is dialect_signal
    assert response_generator.last_prompt_input.dialect_signal is dialect_signal
    assert result.prompt_input.dialect_signal is dialect_signal
    assert result.normalization_result.dialect_label == "Hejazi"
    assert result.code_switch_result.dialect_label == "Hejazi"
    assert result.code_switch_metrics.matrix_language == "AR"
