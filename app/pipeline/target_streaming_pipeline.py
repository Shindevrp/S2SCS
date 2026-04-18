from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.cs_detection.cs_features import CodeSwitchMetrics, analyze_code_switch, whitespace_tokenize
from app.cs_detection.xlmr_model import CodeSwitchResult
from app.dialect.camel_dialect import DialectSignal
from app.llm.prompt_builder import ResponsePromptInput
from app.normalization.arabic_normalizer import NormalizationResult
from app.utils.logger import get_logger


@dataclass
class StreamingTextUpdate:
    update_index: int
    partial_transcript: str
    normalization_result: NormalizationResult
    dialect_signal: DialectSignal
    code_switch_result: CodeSwitchResult
    code_switch_metrics: CodeSwitchMetrics
    sliding_window_text: str


@dataclass
class StreamingTurnResponse:
    update: StreamingTextUpdate
    prompt_input: ResponsePromptInput
    llm_response: Any


@dataclass
class ContextBuffer:
    max_updates: int = 8
    user_updates: list[StreamingTextUpdate] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)

    def push_update(self, update: StreamingTextUpdate) -> None:
        self.user_updates.append(update)
        if len(self.user_updates) > self.max_updates:
            self.user_updates = self.user_updates[-self.max_updates :]

    def push_assistant_response(self, text: str) -> None:
        normalized = text.strip()
        if not normalized:
            return
        self.assistant_responses.append(normalized)
        if len(self.assistant_responses) > self.max_updates:
            self.assistant_responses = self.assistant_responses[-self.max_updates :]

    def build_instruction_suffix(self) -> str:
        if not self.user_updates:
            return ""

        latest = self.user_updates[-1]
        recent_csi = [update.code_switch_metrics.cs_index for update in self.user_updates]
        average_csi = sum(recent_csi) / len(recent_csi)

        lines = [
            "", 
            "Real-time context buffer:",
            f"- Updates processed this turn: {len(self.user_updates)}",
            f"- Latest matrix language: {latest.code_switch_metrics.matrix_language}",
            f"- Average rolling CSI: {average_csi:.3f}",
            f"- Latest sliding window text: {latest.sliding_window_text}",
        ]

        if self.assistant_responses:
            lines.append(f"- Previous assistant response: {self.assistant_responses[-1]}")

        return "\n".join(lines)


class StreamingReasoningOrchestrator:
    """Incremental text reasoning for streaming STT updates."""

    def __init__(
        self,
        dialect_identifier,
        text_normalizer,
        code_switch_detector,
        response_generator,
        *,
        sliding_window_tokens: int = 48,
        max_context_updates: int = 8,
    ) -> None:
        if sliding_window_tokens < 1:
            raise ValueError("sliding_window_tokens must be at least 1")
        if max_context_updates < 1:
            raise ValueError("max_context_updates must be at least 1")

        self.dialect_identifier = dialect_identifier
        self.text_normalizer = text_normalizer
        self.code_switch_detector = code_switch_detector
        self.response_generator = response_generator
        self.sliding_window_tokens = sliding_window_tokens
        self.context_buffer = ContextBuffer(max_updates=max_context_updates)
        self.logger = get_logger(self.__class__.__name__)

        self._last_partial_transcript = ""
        self._latest_update: StreamingTextUpdate | None = None
        self._update_index = 0

    def process_partial_transcript(self, partial_transcript: str) -> StreamingTextUpdate:
        transcript = partial_transcript.strip()
        if not transcript:
            raise ValueError("partial_transcript must not be empty")

        # Skip redundant compute when STT repeats the same hypothesis.
        if transcript == self._last_partial_transcript and self._latest_update is not None:
            return self._latest_update

        dialect_signal = self.dialect_identifier.identify(transcript)
        normalization_result = self.text_normalizer.normalize(
            transcript,
            dialect_signal=dialect_signal,
        )

        sliding_window_text = self._select_sliding_window_text(
            normalization_result.normalized_text,
        )
        code_switch_result = self.code_switch_detector.predict_tokens(
            sliding_window_text,
            dialect_signal=dialect_signal,
        )
        code_switch_metrics = analyze_code_switch(
            code_switch_result.predictions,
            dialect_label=dialect_signal.conditioning_label,
        )

        self._update_index += 1
        update = StreamingTextUpdate(
            update_index=self._update_index,
            partial_transcript=transcript,
            normalization_result=normalization_result,
            dialect_signal=dialect_signal,
            code_switch_result=code_switch_result,
            code_switch_metrics=code_switch_metrics,
            sliding_window_text=sliding_window_text,
        )

        self._last_partial_transcript = transcript
        self._latest_update = update
        self.context_buffer.push_update(update)

        self.logger.debug(
            "update=%s chars=%s window_tokens=%s csi=%.3f",
            update.update_index,
            len(transcript),
            len(whitespace_tokenize(sliding_window_text)),
            code_switch_metrics.cs_index,
        )

        return update

    def generate_turn_response(
        self,
        task_instruction: str = "Generate a natural conversational response.",
    ) -> StreamingTurnResponse:
        prompt_input = self.build_turn_prompt_input(task_instruction=task_instruction)
        llm_response = self.response_generator.generate_response(prompt_input)

        return StreamingTurnResponse(
            update=self._latest_update,
            prompt_input=prompt_input,
            llm_response=llm_response,
        )

    def build_turn_prompt_input(
        self,
        task_instruction: str = "Generate a natural conversational response.",
    ) -> ResponsePromptInput:
        if self._latest_update is None:
            raise RuntimeError("No streaming updates available for response generation")

        full_instruction = task_instruction + self.context_buffer.build_instruction_suffix()
        return ResponsePromptInput(
            normalized_text=self._latest_update.normalization_result.normalized_text,
            dialect_signal=self._latest_update.dialect_signal,
            code_switch_metrics=self._latest_update.code_switch_metrics,
            task_instruction=full_instruction,
        )

    def commit_assistant_response(self, response_text: str) -> None:
        self.context_buffer.push_assistant_response(response_text)

    def reset_turn(self) -> None:
        self._last_partial_transcript = ""
        self._latest_update = None
        self._update_index = 0
        self.context_buffer.user_updates.clear()

    def _select_sliding_window_text(self, normalized_text: str) -> str:
        words = whitespace_tokenize(normalized_text)
        if len(words) <= self.sliding_window_tokens:
            return normalized_text

        start_char = words[-self.sliding_window_tokens].start_char
        return normalized_text[start_char:].strip()
