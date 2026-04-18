import torch

from app.tts.tts_model import (
    QwenOmniTTSSynthesizer,
    TTSSynthesisRequest,
    TTSVoiceRegistry,
)
from app.tts.tts_router import ResponseRoutingResult, RoutedTTSSegment


class FakeBatchFeature(dict):
    def to(self, device):
        moved = FakeBatchFeature()
        for key, value in self.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved


class FakeProcessor:
    def __init__(self) -> None:
        self.last_conversations = []

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        self.last_conversations.append(conversation)
        return FakeBatchFeature(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )


class FakeSpeakerModel:
    def __init__(self) -> None:
        self.config = type("Config", (), {"audio_sampling_rate": 24000})()
        self.calls = []
        self.device = "cpu"

    def generate(self, **kwargs):
        if "speaker" in kwargs:
            self.calls.append(("speaker", kwargs["speaker"]))
            raise TypeError("speaker keyword not supported")
        if "spk" in kwargs:
            self.calls.append(("spk", kwargs["spk"]))
            if kwargs["spk"] == "Chelsie":
                return torch.tensor([[9, 9]]), torch.tensor([0.1, 0.2, 0.3])
            return torch.tensor([[8, 8]]), torch.tensor([0.4, 0.5])
        self.calls.append(("none", None))
        return torch.tensor([[7, 7]]), torch.tensor([0.9])


def build_routing_result() -> ResponseRoutingResult:
    registry = TTSVoiceRegistry()
    return ResponseRoutingResult(
        text="ياهلا hello",
        dialect_label="Gulf",
        matrix_language="AR",
        segments=[
            RoutedTTSSegment(
                text="ياهلا",
                language="AR",
                voice=registry.resolve("AR", "Gulf"),
                start_char=0,
                end_char=5,
                token_start_index=0,
                token_end_index=0,
            ),
            RoutedTTSSegment(
                text="hello",
                language="EN",
                voice=registry.resolve("EN"),
                start_char=6,
                end_char=11,
                token_start_index=1,
                token_end_index=1,
            ),
        ],
        token_predictions=[],
    )


def test_synthesize_segment_uses_voice_speaker_with_spk_fallback() -> None:
    synthesizer = QwenOmniTTSSynthesizer(
        processor=FakeProcessor(),
        model=FakeSpeakerModel(),
        device="cpu",
    )
    registry = TTSVoiceRegistry()

    segment = synthesizer.synthesize_segment(
        request=TTSSynthesisRequest(
            text="ياهلا",
            voice=registry.resolve("AR", "Gulf"),
            segment_index=0,
            start_char=0,
            end_char=5,
        )
    )

    assert segment.sample_rate == 24000
    assert segment.waveform.tolist() == [0.1, 0.2, 0.3]
    assert synthesizer.model.calls[0] == ("speaker", "Chelsie")
    assert synthesizer.model.calls[1] == ("spk", "Chelsie")


def test_synthesize_routed_response_concatenates_bilingual_segments() -> None:
    synthesizer = QwenOmniTTSSynthesizer(
        processor=FakeProcessor(),
        model=FakeSpeakerModel(),
        device="cpu",
        pause_duration_ms=100,
    )

    result = synthesizer.synthesize_routed_response(build_routing_result())

    assert result.sample_rate == 24000
    assert len(result.segments) == 2
    assert result.segments[0].voice.voice_id == "ar_gulf_default"
    assert result.segments[1].voice.voice_id == "en_default"
    assert result.segments[0].waveform.tolist() == [0.1, 0.2, 0.3]
    assert result.segments[1].waveform.tolist() == [0.4, 0.5]
    expected_pause = int(24000 * 0.1)
    assert result.waveform.numel() == 3 + expected_pause + 2
