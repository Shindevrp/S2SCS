from app.dialect.camel_dialect import DialectSignal
from app.normalization.arabic_normalizer import ArabicTextNormalizer


def build_signal(label: str) -> DialectSignal:
    return DialectSignal(
        conditioning_label=label,
        confidence=0.9,
        raw_label=label if label == "MSA" else "JED",
        raw_city=None,
        raw_country=None,
        raw_region=None,
        normalized_text="",
        bucket_scores={"MSA": 0.1, "Gulf": 0.1, "Hejazi": 0.8},
        is_fallback=False,
    )


def test_normalize_removes_diacritics_and_normalizes_alef() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("إِنَّ هٰذَا آلْكِتَابَ كَبِيرٌ", build_signal("MSA"))

    assert result.normalized_text == "ان هذا الكتاب كبير"
    assert "remove_diacritics" in result.applied_rules
    assert "normalize_alef_and_hamza" in result.applied_rules


def test_normalize_preserves_english_and_normalizes_common_arabizi() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("hello mar7aba team", build_signal("MSA"))

    assert result.normalized_text == "hello مرحبا team"
    assert "dialect_arabizi_map:msa" in result.applied_rules


def test_normalize_uses_gulf_signal_for_gulf_arabizi_words() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("wesh ga3d", build_signal("Gulf"))

    assert result.normalized_text == "وش قاعد"
    assert "dialect_arabizi_map:gulf" in result.applied_rules


def test_normalize_uses_hejazi_signal_for_hejazi_arabizi_words() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("eish da7een", build_signal("Hejazi"))

    assert result.normalized_text == "ايش دحين"
    assert "dialect_arabizi_map:hejazi" in result.applied_rules


def test_normalize_does_not_force_gulf_arabizi_when_signal_is_msa() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("wesh gabel", build_signal("MSA"))

    assert result.normalized_text == "wesh gabel"


def test_normalize_handles_mixed_arabic_and_arabizi_in_same_token() -> None:
    normalizer = ArabicTextNormalizer()

    result = normalizer.normalize("7بيبي إزيك", build_signal("MSA"))

    assert result.normalized_text == "حبيبي ازيك"
    assert "generic_arabizi_transliteration" in result.applied_rules


def test_normalize_requires_dialect_signal() -> None:
    normalizer = ArabicTextNormalizer()

    try:
        normalizer.normalize("مرحبا", None)
    except ValueError as exc:
        assert "dialect_signal is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when dialect_signal is missing")
