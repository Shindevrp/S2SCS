import pytest

from app.tts.tts_model import TTSVoiceRegistry


def test_voice_registry_routes_arabic_by_dialect() -> None:
    registry = TTSVoiceRegistry()

    assert registry.resolve("AR", "MSA").voice_id == "ar_msa_default"
    assert registry.resolve("AR", "Gulf").voice_id == "ar_gulf_default"
    assert registry.resolve("AR", "Hejazi").voice_id == "ar_hejazi_default"


def test_voice_registry_routes_english_to_default_voice() -> None:
    registry = TTSVoiceRegistry()

    voice = registry.resolve("EN", "Hejazi")

    assert voice.voice_id == "en_default"
    assert voice.language == "EN"
    assert voice.dialect_label is None
    assert voice.speaker_name == "Ethan"


def test_voice_registry_rejects_unsupported_language() -> None:
    registry = TTSVoiceRegistry()

    with pytest.raises(ValueError, match="Unsupported TTS language"):
        registry.resolve("OTHER")
