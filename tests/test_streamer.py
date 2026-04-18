import torch

from app.streaming.streamer import LowLatencyAudioStreamer
from app.tts.tts_model import SynthesizedSegment, TTSVoiceRegistry
from app.tts.tts_router import ResponseRoutingResult, RoutedTTSSegment


class FakeSynthesizer:
    def __init__(self, sample_rate: int = 1000) -> None:
        self.sample_rate = sample_rate
        self.calls = []
        self.registry = TTSVoiceRegistry()
        self.waveforms = {
            0: torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32),
            1: torch.tensor([1.0, 1.1, 1.2, 1.3], dtype=torch.float32),
        }

    def synthesize_segment(self, request):
        self.calls.append(request.segment_index)
        return SynthesizedSegment(
            text=request.text,
            voice=request.voice,
            waveform=self.waveforms[request.segment_index],
            sample_rate=self.sample_rate,
            segment_index=request.segment_index,
            start_char=request.start_char,
            end_char=request.end_char,
        )


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


def test_stream_waveform_uses_small_first_chunk_for_lower_initial_latency() -> None:
    streamer = LowLatencyAudioStreamer(
        chunk_duration_ms=4,
        first_chunk_duration_ms=2,
    )

    chunks = list(
        streamer.stream_waveform(
            waveform=torch.arange(10, dtype=torch.float32),
            sample_rate=1000,
            segment_index=0,
            text="hello",
            voice_id="en_default",
        )
    )

    assert [chunk.waveform.numel() for chunk in chunks] == [2, 4, 4]
    assert chunks[0].is_first_chunk is True
    assert chunks[-1].is_last_chunk is True
    assert chunks[-1].is_stream_end is True
    assert [(chunk.stream_start_sample, chunk.stream_end_sample) for chunk in chunks] == [
        (0, 2),
        (2, 6),
        (6, 10),
    ]


def test_stream_routed_response_synthesizes_segments_lazily() -> None:
    synthesizer = FakeSynthesizer()
    streamer = LowLatencyAudioStreamer(
        synthesizer=synthesizer,
        chunk_duration_ms=4,
        first_chunk_duration_ms=2,
    )

    generator = streamer.stream_routed_response(build_routing_result())

    first_chunk = next(generator)
    assert synthesizer.calls == [0]
    assert first_chunk.segment_index == 0
    assert first_chunk.waveform.numel() == 2

    second_chunk = next(generator)
    assert synthesizer.calls == [0]
    assert second_chunk.segment_index == 0
    assert second_chunk.waveform.numel() == 4

    third_chunk = next(generator)
    assert synthesizer.calls == [0, 1]
    assert third_chunk.segment_index == 1
    assert third_chunk.waveform.numel() == 2


def test_stream_routed_response_can_insert_pause_chunks_between_segments() -> None:
    synthesizer = FakeSynthesizer()
    streamer = LowLatencyAudioStreamer(
        synthesizer=synthesizer,
        chunk_duration_ms=4,
        first_chunk_duration_ms=2,
        inter_segment_pause_ms=2,
    )

    chunks = list(streamer.stream_routed_response(build_routing_result()))

    pause_chunks = [chunk for chunk in chunks if chunk.segment_index is None]

    assert len(pause_chunks) == 1
    assert pause_chunks[0].waveform.numel() == 2
    assert torch.allclose(pause_chunks[0].waveform, torch.zeros(2))
    assert pause_chunks[0].voice_id is None
