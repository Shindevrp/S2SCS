import pytest
import torch

from app.stt.asr_model import ASRAudioChunk, QwBaseerASR


class FakeProcessorOutput(dict):
    def to(self, device: str):
        moved = FakeProcessorOutput()
        for key, value in self.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved


class FakeProcessor:
    def __init__(self, decoded_text: str) -> None:
        self.decoded_text = decoded_text
        self.last_text = None
        self.last_audio = None
        self.last_sampling_rate = None

    def __call__(self, text, audio, sampling_rate, return_tensors):
        self.last_text = text
        self.last_audio = audio
        self.last_sampling_rate = sampling_rate
        return FakeProcessorOutput({"input_features": torch.ones(1, 4)})

    def batch_decode(self, generated_ids, skip_special_tokens):
        return [self.decoded_text]


class FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def to(self, device: str):
        return self

    def eval(self):
        return self


def build_asr(decoded_text: str) -> QwBaseerASR:
    return QwBaseerASR(
        processor=FakeProcessor(decoded_text=decoded_text),
        model=FakeModel(),
        device="cpu",
        local_files_only=True,
    )


def test_transcribe_chunk_returns_single_fallback_segment() -> None:
    asr = build_asr(decoded_text="hello مرحبا")
    chunk = ASRAudioChunk(
        samples=torch.ones(32000, dtype=torch.float32),
        sample_rate=16000,
        chunk_id="chunk-1",
        chunk_start_sec=5.0,
    )

    result = asr.transcribe_chunk(chunk, language_hint="ar-en")

    assert result.text == "hello مرحبا"
    assert len(result.segments) == 1
    assert result.segments[0].start_sec == pytest.approx(5.0)
    assert result.segments[0].end_sec == pytest.approx(7.0)
    assert result.segments[0].text == "hello مرحبا"


def test_transcribe_chunk_parses_bracket_timestamps() -> None:
    asr = build_asr(decoded_text="[0.00-0.80] hello [0.80-1.60] مرحبا")
    chunk = ASRAudioChunk(
        samples=torch.ones(25600, dtype=torch.float32),
        sample_rate=16000,
        chunk_start_sec=10.0,
    )

    result = asr.transcribe_chunk(chunk)

    assert [segment.text for segment in result.segments] == ["hello", "مرحبا"]
    assert result.segments[0].start_sec == pytest.approx(10.0)
    assert result.segments[0].end_sec == pytest.approx(10.8)
    assert result.segments[1].start_sec == pytest.approx(10.8)
    assert result.segments[1].end_sec == pytest.approx(11.6)


def test_transcribe_chunk_parses_json_segments() -> None:
    asr = build_asr(
        decoded_text=(
            '[{"start": 0.0, "end": 0.5, "text": "hello"}, '
            '{"start": 0.5, "end": 1.2, "text": "اهلا"}]'
        )
    )
    chunk = ASRAudioChunk(samples=torch.ones(19200), sample_rate=16000, chunk_start_sec=2.0)

    result = asr.transcribe_chunk(chunk)

    assert result.text == "hello اهلا"
    assert len(result.segments) == 2
    assert result.segments[0].start_sec == pytest.approx(2.0)
    assert result.segments[1].end_sec == pytest.approx(3.2)


def test_transcribe_chunk_rejects_non_16khz_audio() -> None:
    asr = build_asr(decoded_text="ignored")
    chunk = ASRAudioChunk(samples=torch.ones(1000), sample_rate=8000)

    with pytest.raises(ValueError, match="16 kHz"):
        asr.transcribe_chunk(chunk)


def test_transcribe_chunk_rejects_empty_audio() -> None:
    asr = build_asr(decoded_text="ignored")
    chunk = ASRAudioChunk(samples=torch.tensor([]), sample_rate=16000)

    with pytest.raises(ValueError, match="must not be empty"):
        asr.transcribe_chunk(chunk)
