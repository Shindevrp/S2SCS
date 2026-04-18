from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.cs_detection.cs_features import CodeSwitchMetrics, analyze_code_switch
from app.cs_detection.xlmr_model import CodeSwitchResult
from app.dialect.camel_dialect import DialectSignal
from app.llm.prompt_builder import ResponsePromptInput
from app.normalization.arabic_normalizer import NormalizationResult
from app.utils.logger import get_logger


@dataclass
class Stage3ConditionedTextResult:
    original_text: str
    dialect_signal: DialectSignal
    normalization_result: NormalizationResult
    code_switch_result: CodeSwitchResult
    code_switch_metrics: CodeSwitchMetrics
    prompt_input: ResponsePromptInput
    llm_response: Any


class Stage3ConditionedTextPipeline:
    """Text-stage pipeline where Stage 3 dialect explicitly conditions Stages 4, 5, and 7."""

    def __init__(
        self,
        dialect_identifier,
        text_normalizer,
        code_switch_detector,
        response_generator,
    ) -> None:
        self.dialect_identifier = dialect_identifier
        self.text_normalizer = text_normalizer
        self.code_switch_detector = code_switch_detector
        self.response_generator = response_generator
        self.logger = get_logger(self.__class__.__name__)

    def generate_response(
        self,
        transcript_text: str,
        task_instruction: str = "Generate a natural conversational response.",
    ) -> Stage3ConditionedTextResult:
        if not transcript_text or not transcript_text.strip():
            raise ValueError("transcript_text must not be empty")

        dialect_signal = self.dialect_identifier.identify(transcript_text)

        # Stage 4 explicitly consumes Stage 3 dialect output.
        normalization_result = self.text_normalizer.normalize(
            transcript_text,
            dialect_signal=dialect_signal,
        )

        # Stage 5 explicitly consumes the same Stage 3 dialect output.
        code_switch_result = self.code_switch_detector.predict_tokens(
            normalization_result.normalized_text,
            dialect_signal=dialect_signal,
        )
        code_switch_metrics = analyze_code_switch(
            code_switch_result.predictions,
            dialect_label=dialect_signal.conditioning_label,
        )

        # Stage 7 explicitly consumes the same Stage 3 dialect output.
        prompt_input = ResponsePromptInput(
            normalized_text=normalization_result.normalized_text,
            dialect_signal=dialect_signal,
            code_switch_metrics=code_switch_metrics,
            task_instruction=task_instruction,
        )
        llm_response = self.response_generator.generate_response(prompt_input)

        self.logger.debug(
            "dialect=%s normalized_chars=%s tokens=%s csi=%.3f",
            dialect_signal.conditioning_label,
            len(normalization_result.normalized_text),
            len(code_switch_result.predictions),
            code_switch_metrics.cs_index,
        )

        return Stage3ConditionedTextResult(
            original_text=transcript_text,
            dialect_signal=dialect_signal,
            normalization_result=normalization_result,
            code_switch_result=code_switch_result,
            code_switch_metrics=code_switch_metrics,
            prompt_input=prompt_input,
            llm_response=llm_response,
        )
