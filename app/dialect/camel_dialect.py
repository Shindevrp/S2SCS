from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.utils.logger import get_logger


ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
ARABIC_TEXT_PATTERN = re.compile(r"[\u0600-\u06FF\s]+")
DEFAULT_CONDITIONING_LABEL = "MSA"
SUPPORTED_CONDITIONING_LABELS = ("MSA", "Gulf", "Hejazi")
HEJAZI_LABELS = {"JED"}
GULF_LABELS = {"RIY", "DOH", "MUS"}


LABEL_METADATA = {
    "ALE": {"city": "Aleppo", "country": "Syria", "region": "Levant"},
    "ALG": {"city": "Algiers", "country": "Algeria", "region": "Maghreb"},
    "ALX": {"city": "Alexandria", "country": "Egypt", "region": "Nile Basin"},
    "AMM": {"city": "Amman", "country": "Jordan", "region": "Levant"},
    "ASW": {"city": "Aswan", "country": "Egypt", "region": "Nile Basin"},
    "BAG": {"city": "Baghdad", "country": "Iraq", "region": "Mesopotamia"},
    "BAS": {"city": "Basra", "country": "Iraq", "region": "Mesopotamia"},
    "BEI": {"city": "Beirut", "country": "Lebanon", "region": "Levant"},
    "BEN": {"city": "Benghazi", "country": "Libya", "region": "Maghreb"},
    "CAI": {"city": "Cairo", "country": "Egypt", "region": "Nile Basin"},
    "DAM": {"city": "Damascus", "country": "Syria", "region": "Levant"},
    "DOH": {"city": "Doha", "country": "Qatar", "region": "Gulf"},
    "FES": {"city": "Fes", "country": "Morocco", "region": "Maghreb"},
    "JED": {"city": "Jeddah", "country": "Saudi Arabia", "region": "Gulf"},
    "JER": {"city": "Jerusalem", "country": "Palestine", "region": "Levant"},
    "KHA": {"city": "Khartoum", "country": "Sudan", "region": "Nile Basin"},
    "MOS": {"city": "Mosul", "country": "Iraq", "region": "Mesopotamia"},
    "MSA": {"city": None, "country": None, "region": "Modern Standard Arabic"},
    "MUS": {"city": "Muscat", "country": "Oman", "region": "Gulf"},
    "RAB": {"city": "Rabat", "country": "Morocco", "region": "Maghreb"},
    "RIY": {"city": "Riyadh", "country": "Saudi Arabia", "region": "Gulf"},
    "SAL": {"city": "Salt", "country": "Jordan", "region": "Levant"},
    "SAN": {"city": "Sanaa", "country": "Yemen", "region": "Gulf of Aden"},
    "SFX": {"city": "Sfax", "country": "Tunisia", "region": "Maghreb"},
    "TRI": {"city": "Tripoli", "country": "Libya", "region": "Maghreb"},
    "TUN": {"city": "Tunis", "country": "Tunisia", "region": "Maghreb"},
}


@dataclass
class DialectSignal:
    conditioning_label: str
    confidence: float
    raw_label: str
    raw_city: Optional[str]
    raw_country: Optional[str]
    raw_region: Optional[str]
    normalized_text: str
    bucket_scores: dict[str, float]
    is_fallback: bool
    fallback_reason: Optional[str] = None


class CamelDialectIdentifier:
    """Dialect identification wrapper used only as a conditioning signal."""

    def __init__(
        self,
        confidence_threshold: float = 0.40,
        minimum_arabic_chars: int = 6,
        model: Optional[Any] = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if minimum_arabic_chars < 1:
            raise ValueError("minimum_arabic_chars must be at least 1")

        self.confidence_threshold = confidence_threshold
        self.minimum_arabic_chars = minimum_arabic_chars
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or self._load_model()

    def identify(self, text: str) -> DialectSignal:
        normalized_text = self._extract_arabic_text(text)

        if self._arabic_char_count(normalized_text) < self.minimum_arabic_chars:
            self.logger.debug("Insufficient Arabic content for dialect conditioning")
            return self._fallback_signal(
                normalized_text=normalized_text,
                fallback_reason="insufficient_arabic_content",
            )

        try:
            prediction = self.model.predict([normalized_text], output="label")[0]
        except Exception as exc:
            self.logger.exception("Dialect identification failed")
            raise RuntimeError("CAMeL dialect identification failed") from exc

        raw_label = str(prediction.top)
        raw_scores = {str(label): float(score) for label, score in prediction.scores.items()}
        confidence = raw_scores.get(raw_label, 0.0)
        bucket_scores = self._aggregate_bucket_scores(raw_scores)
        metadata = LABEL_METADATA.get(raw_label, {})
        conditioning_label = self._map_to_conditioning_label(raw_label)

        if conditioning_label is None:
            return self._fallback_signal(
                normalized_text=normalized_text,
                raw_label=raw_label,
                raw_scores=raw_scores,
                bucket_scores=bucket_scores,
                fallback_reason="out_of_scope_dialect",
            )

        if confidence < self.confidence_threshold and conditioning_label != DEFAULT_CONDITIONING_LABEL:
            return self._fallback_signal(
                normalized_text=normalized_text,
                raw_label=raw_label,
                raw_scores=raw_scores,
                bucket_scores=bucket_scores,
                fallback_reason="low_confidence",
            )

        signal = DialectSignal(
            conditioning_label=conditioning_label,
            confidence=confidence,
            raw_label=raw_label,
            raw_city=metadata.get("city"),
            raw_country=metadata.get("country"),
            raw_region=metadata.get("region"),
            normalized_text=normalized_text,
            bucket_scores=bucket_scores,
            is_fallback=False,
        )

        self.logger.debug(
            "raw_label=%s conditioning_label=%s confidence=%.4f",
            raw_label,
            conditioning_label,
            confidence,
        )
        return signal

    def classify_label(self, text: str) -> str:
        """Return only one strict label: MSA, Gulf, or Hejazi."""
        signal = self.identify(text)
        label = signal.conditioning_label
        if label not in SUPPORTED_CONDITIONING_LABELS:
            return DEFAULT_CONDITIONING_LABEL
        return label

    def classify_json(self, text: str) -> dict[str, object]:
        """Return pipeline-friendly dialect JSON payload."""
        signal = self.identify(text)
        confidence = max(0.0, min(1.0, float(signal.confidence)))

        if signal.is_fallback:
            reason = f"fallback: {signal.fallback_reason or 'unspecified'}"
        else:
            reason = f"raw={signal.raw_label} mapped={signal.conditioning_label}"

        return {
            "dialect": signal.conditioning_label,
            "confidence": confidence,
            "reason": reason,
        }

    def _load_model(self) -> Any:
        try:
            from camel_tools.dialectid import DialectIdentifier
        except ImportError as exc:
            self.logger.exception("camel-tools is not installed")
            raise RuntimeError(
                "camel-tools is required. Install it with `pip install camel-tools` "
                "and download the dialect model with `camel_data -i dialectid-default`."
            ) from exc

        try:
            return DialectIdentifier.pretrained()
        except Exception as exc:
            self.logger.exception("Failed to load CAMeL dialect model")
            raise RuntimeError(
                "Failed to load CAMeL dialect model. Ensure `camel_data -i dialectid-default` "
                "has been run and CAMELTOOLS_DATA is set if using a custom offline path."
            ) from exc

    def _extract_arabic_text(self, text: str) -> str:
        matches = ARABIC_TEXT_PATTERN.findall(text or "")
        normalized = " ".join(part.strip() for part in matches if part.strip())
        return re.sub(r"\s+", " ", normalized).strip()

    def _arabic_char_count(self, text: str) -> int:
        return len(ARABIC_CHAR_PATTERN.findall(text))

    def _aggregate_bucket_scores(self, raw_scores: dict[str, float]) -> dict[str, float]:
        bucket_scores = {label: 0.0 for label in SUPPORTED_CONDITIONING_LABELS}
        for raw_label, score in raw_scores.items():
            mapped_label = self._map_to_conditioning_label(raw_label)
            if mapped_label is not None:
                bucket_scores[mapped_label] += float(score)
        return bucket_scores

    def _map_to_conditioning_label(self, raw_label: str) -> Optional[str]:
        if raw_label == "MSA":
            return "MSA"
        if raw_label in HEJAZI_LABELS:
            return "Hejazi"
        if raw_label in GULF_LABELS:
            return "Gulf"
        return None

    def _fallback_signal(
        self,
        normalized_text: str,
        fallback_reason: str,
        raw_label: str = "MSA",
        raw_scores: Optional[dict[str, float]] = None,
        bucket_scores: Optional[dict[str, float]] = None,
    ) -> DialectSignal:
        scores = bucket_scores or {label: 0.0 for label in SUPPORTED_CONDITIONING_LABELS}
        if raw_scores and raw_label in raw_scores:
            confidence = raw_scores[raw_label]
        else:
            confidence = scores.get(DEFAULT_CONDITIONING_LABEL, 0.0)

        metadata = LABEL_METADATA.get(raw_label, LABEL_METADATA["MSA"])
        self.logger.debug("Falling back to MSA conditioning: %s", fallback_reason)

        return DialectSignal(
            conditioning_label=DEFAULT_CONDITIONING_LABEL,
            confidence=confidence,
            raw_label=raw_label,
            raw_city=metadata.get("city"),
            raw_country=metadata.get("country"),
            raw_region=metadata.get("region"),
            normalized_text=normalized_text,
            bucket_scores=scores,
            is_fallback=True,
            fallback_reason=fallback_reason,
        )
