from __future__ import annotations

from dataclasses import dataclass

from app.cs_detection.cs_features import CodeSwitchMetrics
from app.dialect.camel_dialect import DialectSignal


@dataclass
class ResponsePromptInput:
    normalized_text: str
    dialect_signal: DialectSignal
    code_switch_metrics: CodeSwitchMetrics
    task_instruction: str = "Generate a natural conversational response."


class CodeSwitchPromptBuilder:
    """Builds a structured prompt for bilingual response generation."""

    def build(self, prompt_input: ResponsePromptInput) -> str:
        if prompt_input.dialect_signal is None:
            raise ValueError(
                "dialect_signal is required because Stage 7 prompting must be conditioned "
                "on Stage 3 dialect identification."
            )

        islands_text = self._format_islands(prompt_input.code_switch_metrics)
        matrix_language_name = self._expand_language(prompt_input.code_switch_metrics.matrix_language)
        secondary_language_name = self._expand_language(
            prompt_input.code_switch_metrics.secondary_language
        )
        dialect_label = prompt_input.dialect_signal.conditioning_label
        dialect_confidence = prompt_input.dialect_signal.confidence
        dialect_fallback = (
            prompt_input.dialect_signal.fallback_reason
            if prompt_input.dialect_signal.is_fallback
            else "none"
        )

        return (
            "You are a bilingual Arabic-English conversational assistant for speech-to-speech.\n"
            "Generate one user-facing reply that preserves the speaker's code-switching style.\n"
            "Do not flatten the response into only Arabic or only English.\n"
            "Keep the same dialectal flavor when Arabic is used.\n"
            "Preserve named entities and natural spoken tone.\n"
            "Output only the final response text.\n\n"
            "Task:\n"
            f"{prompt_input.task_instruction}\n\n"
            "Conditioning Signals:\n"
            f"- Normalized user text: {prompt_input.normalized_text}\n"
            f"- Arabic dialect signal: {dialect_label}\n"
            f"- Dialect confidence: {dialect_confidence:.3f}\n"
            f"- Dialect fallback reason: {dialect_fallback}\n"
            f"- Code-switch index (CSI): {prompt_input.code_switch_metrics.cs_index:.3f}\n"
            f"- Matrix language: {matrix_language_name}\n"
            f"- Secondary language: {secondary_language_name}\n"
            f"- Embedded language islands: {islands_text}\n\n"
            "Style Constraints:\n"
            "- Match the user's switching pattern and overall language balance.\n"
            "- If CSI is low, keep switching light and natural.\n"
            "- If CSI is high, allow more frequent natural switching.\n"
            "- Respect the matrix language as the dominant language of the reply.\n"
            f"- When Arabic appears, make it sound compatible with {dialect_label} usage.\n"
            "- Keep the response concise and speech-friendly.\n"
        )

    def _format_islands(self, metrics: CodeSwitchMetrics) -> str:
        if not metrics.embedded_language_islands:
            return "none"

        return "; ".join(
            f"{island.language}: {island.text}"
            for island in metrics.embedded_language_islands
        )

    def _expand_language(self, label: str | None) -> str:
        if label == "AR":
            return "Arabic"
        if label == "EN":
            return "English"
        if label is None:
            return "none"
        return label
