from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.cs_detection.cs_features import TokenPrediction, analyze_code_switch
from app.cs_detection.xlmr_model import XLMRCodeSwitchDetector
from app.dialect.camel_dialect import DEFAULT_CONDITIONING_LABEL, DialectSignal
from app.tts.tts_model import TTSVoiceProfile, TTSVoiceRegistry
from app.utils.logger import get_logger


@dataclass
class RoutedTTSSegment:
    text: str
    language: str
    voice: TTSVoiceProfile
    start_char: int
    end_char: int
    token_start_index: int
    token_end_index: int


@dataclass
class ResponseRoutingResult:
    text: str
    dialect_label: str
    matrix_language: str
    segments: list[RoutedTTSSegment]
    token_predictions: list[TokenPrediction]


class ResponseTTSRouter:
    """Analyze a generated response, segment by language, and route each segment to a TTS voice."""

    def __init__(
        self,
        code_switch_detector: Optional[XLMRCodeSwitchDetector] = None,
        voice_registry: Optional[TTSVoiceRegistry] = None,
    ) -> None:
        self.code_switch_detector = code_switch_detector
        self.voice_registry = voice_registry or TTSVoiceRegistry()
        self.logger = get_logger(self.__class__.__name__)

    def route_response(
        self,
        text: str,
        dialect_signal: Optional[DialectSignal] = None,
        token_predictions: Optional[list[TokenPrediction]] = None,
    ) -> ResponseRoutingResult:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        predictions = token_predictions or self._detect_tokens(text, dialect_signal)
        metrics = analyze_code_switch(
            predictions=predictions,
            dialect_label=self._resolve_dialect_label(dialect_signal, predictions),
        )

        spoken_labels = [
            self._resolve_spoken_language(
                index=index,
                predictions=predictions,
                matrix_language=metrics.matrix_language,
            )
            for index in range(len(predictions))
        ]

        segments = self._build_segments(
            text=text,
            predictions=predictions,
            spoken_labels=spoken_labels,
            dialect_label=metrics.dialect_label,
        )

        self.logger.debug(
            "matrix_language=%s segments=%s predictions=%s",
            metrics.matrix_language,
            len(segments),
            len(predictions),
        )

        return ResponseRoutingResult(
            text=text,
            dialect_label=metrics.dialect_label,
            matrix_language=metrics.matrix_language,
            segments=segments,
            token_predictions=predictions,
        )

    def _detect_tokens(
        self,
        text: str,
        dialect_signal: Optional[DialectSignal],
    ) -> list[TokenPrediction]:
        if self.code_switch_detector is None:
            raise RuntimeError(
                "No code-switch detector was provided. "
                "Pass token_predictions directly or initialize ResponseTTSRouter with "
                "an XLMRCodeSwitchDetector."
            )
        if dialect_signal is None:
            raise ValueError(
                "dialect_signal is required when routing through the Stage 5 detector "
                "because code-switch detection is dialect-conditioned."
            )

        detection_result = self.code_switch_detector.predict_tokens(
            text=text,
            dialect_signal=dialect_signal,
        )
        return detection_result.predictions

    def _build_segments(
        self,
        text: str,
        predictions: list[TokenPrediction],
        spoken_labels: list[str],
        dialect_label: str,
    ) -> list[RoutedTTSSegment]:
        if not predictions:
            return []

        segments: list[RoutedTTSSegment] = []
        current_language = spoken_labels[0]
        start_index = 0

        for index in range(1, len(predictions)):
            if spoken_labels[index] == current_language:
                continue

            segments.append(
                self._make_segment(
                    text=text,
                    predictions=predictions,
                    language=current_language,
                    start_index=start_index,
                    end_index=index - 1,
                    dialect_label=dialect_label,
                )
            )
            current_language = spoken_labels[index]
            start_index = index

        segments.append(
            self._make_segment(
                text=text,
                predictions=predictions,
                language=current_language,
                start_index=start_index,
                end_index=len(predictions) - 1,
                dialect_label=dialect_label,
            )
        )

        return segments

    def _make_segment(
        self,
        text: str,
        predictions: list[TokenPrediction],
        language: str,
        start_index: int,
        end_index: int,
        dialect_label: str,
    ) -> RoutedTTSSegment:
        first_token = predictions[start_index]
        last_token = predictions[end_index]
        segment_text = text[first_token.start_char:last_token.end_char].strip()

        voice = self.voice_registry.resolve(
            language=language,
            dialect_label=dialect_label if language == "AR" else None,
        )

        return RoutedTTSSegment(
            text=segment_text,
            language=language,
            voice=voice,
            start_char=first_token.start_char,
            end_char=last_token.end_char,
            token_start_index=start_index,
            token_end_index=end_index,
        )

    def _resolve_spoken_language(
        self,
        index: int,
        predictions: list[TokenPrediction],
        matrix_language: str,
    ) -> str:
        label = predictions[index].label
        if label in {"AR", "EN"}:
            return label

        previous_language = self._nearest_language(predictions, index, step=-1)
        next_language = self._nearest_language(predictions, index, step=1)

        if previous_language and previous_language == next_language:
            return previous_language
        if previous_language:
            return previous_language
        if next_language:
            return next_language
        if matrix_language in {"AR", "EN"}:
            return matrix_language
        return "AR"

    def _nearest_language(
        self,
        predictions: list[TokenPrediction],
        index: int,
        step: int,
    ) -> Optional[str]:
        probe = index + step
        while 0 <= probe < len(predictions):
            label = predictions[probe].label
            if label in {"AR", "EN"}:
                return label
            probe += step
        return None

    def _resolve_dialect_label(
        self,
        dialect_signal: Optional[DialectSignal],
        predictions: list[TokenPrediction],
    ) -> str:
        if dialect_signal is not None and dialect_signal.conditioning_label:
            return dialect_signal.conditioning_label
        if predictions:
            return predictions[0].dialect_label or DEFAULT_CONDITIONING_LABEL
        return DEFAULT_CONDITIONING_LABEL
