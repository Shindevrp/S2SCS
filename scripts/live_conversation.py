from __future__ import annotations

import argparse
from time import perf_counter

from app.audio.capture import MicrophoneTurnCapture
from app.audio.playback import AudioPlayback
from app.config import load_app_config
from app.pipeline.e2e_pipeline import EndToEndSpeechPipeline
from app.utils.logger import get_logger


class LiveConversationRunner:
    def __init__(self, config_path: str | None = None, task_instruction: str | None = None) -> None:
        self.config = load_app_config(config_path)
        self.pipeline = EndToEndSpeechPipeline.from_config(self.config)
        self.capture = MicrophoneTurnCapture(
            vad=self.pipeline.vad,
            sample_rate=self.config.audio.input_sample_rate,
            frame_ms=self.config.audio.frame_ms,
            end_silence_ms=self.config.audio.end_silence_ms,
            min_speech_ms=self.config.audio.min_speech_ms,
            max_utterance_s=self.config.audio.max_utterance_s,
        )
        self.playback = AudioPlayback()
        self.task_instruction = task_instruction or self.config.pipeline.task_instruction
        self.logger = get_logger(self.__class__.__name__)

    def run(self) -> None:
        self.logger.info("Listening for live conversation turns at %s Hz", self.config.audio.input_sample_rate)
        self.logger.info("Press Ctrl+C to stop.")

        for turn in self.capture.iter_turns():
            started = perf_counter()
            try:
                transcription = self.pipeline.transcribe_audio(
                    turn.samples,
                    turn.sample_rate,
                    language_hint=self.config.pipeline.language_hint,
                    apply_vad=False,
                )
                if not transcription.transcript_text:
                    continue

                response = self.pipeline.respond_to_text(
                    transcription.transcript_text,
                    task_instruction=self.task_instruction,
                )
                self.logger.info("User: %s", transcription.transcript_text)
                self.logger.info("Assistant: %s", response.response_text)
                self.playback.play_stream(self.pipeline.iter_audio_chunks(response))
                self.logger.info("Turn latency: %.2fs", perf_counter() - started)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.logger.warning("Live turn failed: %s", exc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven live speech conversation")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--task",
        default=None,
        help="Optional override for the response-generation task instruction",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = LiveConversationRunner(config_path=args.config, task_instruction=args.task)
    app.run()


if __name__ == "__main__":
    main()
