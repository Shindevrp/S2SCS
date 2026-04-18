from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.dialect.camel_dialect import DEFAULT_CONDITIONING_LABEL, DialectSignal
from app.utils.logger import get_logger


DIACRITICS_PATTERN = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF]")
ARABIZI_RUN_PATTERN = re.compile(r"[A-Za-z0-9']+")
WHITESPACE_TOKEN_PATTERN = re.compile(r"\s+|[^\s]+")
EDGE_PUNCTUATION_PATTERN = re.compile(r"^([^A-Za-z0-9\u0600-\u06FF']*)(.*?)([^A-Za-z0-9\u0600-\u06FF']*)$")

ALEF_VARIANTS = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
)


COMMON_ARABIZI_WORDS = {
    "3ndi": "عندي",
    "3shan": "عشان",
    "7abibi": "حبيبي",
    "7abibti": "حبيبتي",
    "7amdulillah": "الحمدلله",
    "ahlan": "اهلا",
    "allah": "الله",
    "ana": "انا",
    "enta": "انت",
    "enti": "انتي",
    "inshallah": "ان شاء الله",
    "ma3a": "مع",
    "mar7aba": "مرحبا",
    "salam": "سلام",
    "ya3ni": "يعني",
}

GULF_ARABIZI_WORDS = {
    "gabel": "قبل",
    "ga3d": "قاعد",
    "hal7een": "الحين",
    "shlon": "شلون",
    "wish": "وش",
    "wesh": "وش",
    "wsh": "وش",
    "7ag": "حق",
    "7aq": "حق",
}

HEJAZI_ARABIZI_WORDS = {
    "da7een": "دحين",
    "d7een": "دحين",
    "eish": "ايش",
    "esh": "ايش",
    "ish": "ايش",
    "lissa": "لسه",
    "mara": "مرة",
}

ARABIC_WORD_NORMALIZATION = {
    "Hejazi": {
        "ايش": "ايش",
        "إيش": "ايش",
        "دحين": "دحين",
        "دحّين": "دحين",
    },
    "Gulf": {
        "الحين": "الحين",
        "وش": "وش",
    },
    "MSA": {},
}

COMMON_SEQUENCE_MAP = [
    ("sh", "ش"),
    ("kh", "خ"),
    ("gh", "غ"),
    ("th", "ث"),
    ("dh", "ذ"),
    ("ch", "تش"),
]

COMMON_CHAR_MAP = {
    "'": "ء",
    "2": "ء",
    "3": "ع",
    "4": "غ",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "ق",
    "9": "ص",
    "a": "ا",
    "b": "ب",
    "d": "د",
    "f": "ف",
    "h": "ه",
    "i": "ي",
    "j": "ج",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "o": "و",
    "p": "ب",
    "q": "ق",
    "r": "ر",
    "s": "س",
    "t": "ت",
    "u": "و",
    "v": "ف",
    "w": "و",
    "x": "كس",
    "y": "ي",
    "z": "ز",
}

DIALECT_CHAR_OVERRIDES = {
    "MSA": {},
    "Gulf": {"g": "ق"},
    "Hejazi": {"g": "ق"},
}


@dataclass
class NormalizationResult:
    original_text: str
    normalized_text: str
    dialect_label: str
    applied_rules: list[str]


class ArabicTextNormalizer:
    """Arabic normalizer that is explicitly conditioned on Stage 3 dialect output."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    def normalize(
        self,
        text: str,
        dialect_signal: DialectSignal,
    ) -> NormalizationResult:
        if dialect_signal is None:
            raise ValueError(
                "dialect_signal is required because Stage 4 normalization must be "
                "conditioned on Stage 3 dialect identification."
            )

        dialect_label = self._resolve_dialect_label(dialect_signal)
        applied_rules: list[str] = []
        normalized_tokens: list[str] = []

        for token in WHITESPACE_TOKEN_PATTERN.findall(text or ""):
            if token.isspace():
                normalized_tokens.append(token)
                continue

            normalized_tokens.append(
                self._normalize_token(
                    token=token,
                    dialect_label=dialect_label,
                    applied_rules=applied_rules,
                )
            )

        normalized_text = "".join(normalized_tokens).strip()
        self.logger.debug(
            "dialect_label=%s original_chars=%s normalized_chars=%s",
            dialect_label,
            len(text or ""),
            len(normalized_text),
        )

        return NormalizationResult(
            original_text=text,
            normalized_text=normalized_text,
            dialect_label=dialect_label,
            applied_rules=self._deduplicate_rules(applied_rules),
        )

    def _normalize_token(
        self,
        token: str,
        dialect_label: str,
        applied_rules: list[str],
    ) -> str:
        match = EDGE_PUNCTUATION_PATTERN.match(token)
        if not match:
            return token

        prefix, body, suffix = match.groups()
        if not body:
            return token

        if ARABIC_CHAR_PATTERN.search(body):
            normalized_body = ARABIZI_RUN_PATTERN.sub(
                lambda run: self._normalize_embedded_run(
                    run.group(0),
                    dialect_label=dialect_label,
                    applied_rules=applied_rules,
                ),
                body,
            )
            normalized_body = self._normalize_arabic_script(
                normalized_body,
                dialect_label=dialect_label,
                applied_rules=applied_rules,
            )
            return f"{prefix}{normalized_body}{suffix}"

        if self._looks_like_arabizi(body, dialect_label=dialect_label):
            normalized_body = self._transliterate_arabizi(
                body,
                dialect_label=dialect_label,
                applied_rules=applied_rules,
            )
            normalized_body = self._normalize_arabic_script(
                normalized_body,
                dialect_label=dialect_label,
                applied_rules=applied_rules,
            )
            return f"{prefix}{normalized_body}{suffix}"

        return token

    def _normalize_embedded_run(
        self,
        run: str,
        dialect_label: str,
        applied_rules: list[str],
    ) -> str:
        if self._looks_like_arabizi(run, dialect_label=dialect_label):
            return self._transliterate_arabizi(
                run,
                dialect_label=dialect_label,
                applied_rules=applied_rules,
            )
        return run

    def _normalize_arabic_script(
        self,
        text: str,
        dialect_label: str,
        applied_rules: list[str],
    ) -> str:
        normalized = text

        stripped = DIACRITICS_PATTERN.sub("", normalized)
        if stripped != normalized:
            applied_rules.append("remove_diacritics")
            normalized = stripped

        without_tatweel = normalized.replace("ـ", "")
        if without_tatweel != normalized:
            applied_rules.append("remove_tatweel")
            normalized = without_tatweel

        unified_alef = normalized.translate(ALEF_VARIANTS)
        if unified_alef != normalized:
            applied_rules.append("normalize_alef_and_hamza")
            normalized = unified_alef

        dialect_map = ARABIC_WORD_NORMALIZATION.get(dialect_label, {})
        if normalized in dialect_map:
            mapped = dialect_map[normalized]
            if mapped != normalized:
                applied_rules.append(f"dialect_arabic_map:{dialect_label.lower()}")
                normalized = mapped

        return normalized

    def _transliterate_arabizi(
        self,
        token: str,
        dialect_label: str,
        applied_rules: list[str],
    ) -> str:
        normalized_token = token.lower()
        lexicon = self._get_arabizi_lexicon(dialect_label)

        if normalized_token in lexicon:
            applied_rules.append(f"dialect_arabizi_map:{dialect_label.lower()}")
            return lexicon[normalized_token]

        transliterated = normalized_token
        for source, target in COMMON_SEQUENCE_MAP:
            if source in transliterated:
                transliterated = transliterated.replace(source, target)

        char_map = dict(COMMON_CHAR_MAP)
        char_map.update(DIALECT_CHAR_OVERRIDES.get(dialect_label, {}))

        pieces: list[str] = []
        for char in transliterated:
            pieces.append(char_map.get(char, char))

        result = "".join(pieces)
        if result != token:
            applied_rules.append("generic_arabizi_transliteration")
        return result

    def _looks_like_arabizi(self, token: str, dialect_label: str) -> bool:
        lowered = token.lower()
        lexicon = self._get_arabizi_lexicon(dialect_label)
        if lowered in lexicon:
            return True
        if re.search(r"[23456789']", lowered):
            return True
        return False

    def _get_arabizi_lexicon(self, dialect_label: str) -> dict[str, str]:
        lexicon = dict(COMMON_ARABIZI_WORDS)
        if dialect_label == "Gulf":
            lexicon.update(GULF_ARABIZI_WORDS)
        elif dialect_label == "Hejazi":
            lexicon.update(HEJAZI_ARABIZI_WORDS)
        return lexicon

    def _resolve_dialect_label(self, dialect_signal: Optional[DialectSignal]) -> str:
        if dialect_signal is None:
            return DEFAULT_CONDITIONING_LABEL
        return dialect_signal.conditioning_label or DEFAULT_CONDITIONING_LABEL

    def _deduplicate_rules(self, rules: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for rule in rules:
            if rule in seen:
                continue
            seen.add(rule)
            ordered.append(rule)
        return ordered
