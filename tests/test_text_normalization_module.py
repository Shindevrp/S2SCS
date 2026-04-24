from app.normalization.arabic_normalizer import (
    convert_arabizi_token,
    is_arabizi_token,
    normalize_text,
)


def test_normalize_text_handles_alef_hamza_and_diacritics() -> None:
    text = "أُؤُ إِئُ آ"

    normalized = normalize_text(text, dialect="MSA")

    assert normalized == "او اي ا"


def test_normalize_text_preserves_english_words() -> None:
    text = "hello world this is a test"

    normalized = normalize_text(text, dialect="Gulf")

    assert normalized == text


def test_arabizi_detection_and_conversion() -> None:
    assert is_arabizi_token("3ndi") is True
    assert is_arabizi_token("hello") is False
    assert convert_arabizi_token("3ndi") == "عndi"
    assert convert_arabizi_token("hello") == "hello"


def test_normalize_text_applies_saudi_slang_for_saudi_dialects() -> None:
    text = "وش يبغى مره ايش"

    normalized = normalize_text(text, dialect="Gulf")

    assert normalized == "ماذا يريد جدا ماذا"


def test_normalize_text_keeps_slang_unchanged_for_msa() -> None:
    text = "وش يبغى مره ايش"

    normalized = normalize_text(text, dialect="MSA")

    assert normalized == "وش يبغى مره ايش"
