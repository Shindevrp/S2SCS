import pytest
import torch

from app.vad.silero_vad import AudioChunk, SileroVAD


class FakeSileroModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.last_audio = None
        self.last_sample_rate = None
        self.seen_lengths = []

    def __call__(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.last_audio = audio
        self.last_sample_rate = sample_rate
        self.seen_lengths.append(int(audio.numel()))
        return torch.tensor(self.probability, dtype=torch.float32)


def test_process_chunk_returns_speech_for_probability_above_threshold() -> None:
    vad = SileroVAD(threshold=0.5, model=FakeSileroModel(probability=0.91))
    chunk = AudioChunk(
        samples=torch.ones(1600, dtype=torch.float32),
        sample_rate=16000,
        chunk_id="chunk-1",
    )

    result = vad.process_chunk(chunk)

    assert result.is_speech is True
    assert result.speech_probability == pytest.approx(0.91)
    assert result.chunk_id == "chunk-1"
    assert result.num_samples == 1600


def test_process_chunk_returns_non_speech_for_probability_below_threshold() -> None:
    vad = SileroVAD(threshold=0.75, model=FakeSileroModel(probability=0.2))
    chunk = AudioChunk(samples=torch.zeros(800), sample_rate=8000)

    result = vad.process_chunk(chunk)

    assert result.is_speech is False
    assert result.speech_probability == pytest.approx(0.2)


def test_process_chunk_converts_multichannel_audio_to_mono() -> None:
    model = FakeSileroModel(probability=0.8)
    vad = SileroVAD(model=model)
    stereo_chunk = AudioChunk(samples=torch.ones(2, 400), sample_rate=16000)

    result = vad.process_chunk(stereo_chunk)

    assert result.num_samples == 400
    assert model.last_audio.ndim == 1
    assert model.last_sample_rate == 16000


def test_process_chunk_rejects_empty_audio() -> None:
    vad = SileroVAD(model=FakeSileroModel(probability=0.1))
    chunk = AudioChunk(samples=torch.tensor([]), sample_rate=16000)

    with pytest.raises(ValueError, match="must not be empty"):
        vad.process_chunk(chunk)


def test_process_chunk_rejects_unsupported_sample_rate() -> None:
    vad = SileroVAD(model=FakeSileroModel(probability=0.1))
    chunk = AudioChunk(samples=torch.ones(160), sample_rate=22050)

    with pytest.raises(ValueError, match="Unsupported sample rate"):
        vad.process_chunk(chunk)


def test_process_chunk_splits_large_16khz_audio_into_silero_windows() -> None:
    model = FakeSileroModel(probability=0.9)
    vad = SileroVAD(model=model)
    chunk = AudioChunk(samples=torch.ones(1600, dtype=torch.float32), sample_rate=16000)

    result = vad.process_chunk(chunk)

    assert result.is_speech is True
    assert result.num_samples == 1600
    assert model.seen_lengths == [512, 512, 512, 512]
