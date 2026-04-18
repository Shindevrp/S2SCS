from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from app.utils.logger import get_logger

if TYPE_CHECKING:
    from app.tts.tts_router import ResponseRoutingResult, RoutedTTSSegment


DEFAULT_QWEN_OMNI_MODEL = "models/Qwen/Qwen2.5-Omni-3B"
SUPPORTED_QWEN_OMNI_MODELS = (
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-Omni-7B",
)
QWEN_OMNI_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


@dataclass(frozen=True)
class TTSVoiceProfile:
    voice_id: str
    language: str
    dialect_label: Optional[str]
    speaker_name: Optional[str] = None
    sample_rate: int = 24000


@dataclass
class TTSSynthesisRequest:
    text: str
    voice: TTSVoiceProfile
    segment_index: int
    start_char: int
    end_char: int


@dataclass
class SynthesizedSegment:
    text: str
    voice: TTSVoiceProfile
    waveform: torch.Tensor
    sample_rate: int
    segment_index: int
    start_char: int
    end_char: int


@dataclass
class BilingualSpeechResult:
    waveform: torch.Tensor
    sample_rate: int
    segments: list[SynthesizedSegment]
    model_name_or_path: str


class TTSVoiceRegistry:
    """Simple voice registry for segment-level routing."""

    def __init__(
        self,
        arabic_msa_voice_id: str = "ar_msa_default",
        arabic_gulf_voice_id: str = "ar_gulf_default",
        arabic_hejazi_voice_id: str = "ar_hejazi_default",
        english_voice_id: str = "en_default",
        arabic_speaker_name: str = "Chelsie",
        english_speaker_name: str = "Ethan",
    ) -> None:
        self.arabic_msa_voice = TTSVoiceProfile(
            voice_id=arabic_msa_voice_id,
            language="AR",
            dialect_label="MSA",
            speaker_name=arabic_speaker_name,
        )
        self.arabic_gulf_voice = TTSVoiceProfile(
            voice_id=arabic_gulf_voice_id,
            language="AR",
            dialect_label="Gulf",
            speaker_name=arabic_speaker_name,
        )
        self.arabic_hejazi_voice = TTSVoiceProfile(
            voice_id=arabic_hejazi_voice_id,
            language="AR",
            dialect_label="Hejazi",
            speaker_name=arabic_speaker_name,
        )
        self.english_voice = TTSVoiceProfile(
            voice_id=english_voice_id,
            language="EN",
            dialect_label=None,
            speaker_name=english_speaker_name,
        )

    def resolve(
        self, language: str, dialect_label: Optional[str] = None
    ) -> TTSVoiceProfile:
        if language == "EN":
            return self.english_voice

        if language != "AR":
            raise ValueError(f"Unsupported TTS language: {language}")

        if dialect_label == "Gulf":
            return self.arabic_gulf_voice
        if dialect_label == "Hejazi":
            return self.arabic_hejazi_voice
        return self.arabic_msa_voice


class QwenOmniTTSSynthesizer:
    """Offline-capable bilingual TTS using Qwen2.5-Omni."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_QWEN_OMNI_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
        pause_duration_ms: int = 120,
        processor: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.local_files_only = local_files_only
        self.pause_duration_ms = pause_duration_ms
        self.logger = get_logger(self.__class__.__name__)

        self.processor = processor
        self.model = model
        if self.processor is None or self.model is None:
            self.processor, self.model = self._load_components()

    def synthesize_routed_response(
        self,
        routing_result: "ResponseRoutingResult",
    ) -> BilingualSpeechResult:
        synthesized_segments: list[SynthesizedSegment] = []
        sample_rate = 24000

        for index, segment in enumerate(routing_result.segments):
            synthesized = self.synthesize_segment(
                request=TTSSynthesisRequest(
                    text=segment.text,
                    voice=segment.voice,
                    segment_index=index,
                    start_char=segment.start_char,
                    end_char=segment.end_char,
                )
            )
            synthesized_segments.append(synthesized)
            sample_rate = synthesized.sample_rate

        full_waveform = self._concatenate_segments(
            synthesized_segments,
            sample_rate=sample_rate,
        )

        self.logger.debug(
            "segments=%s total_samples=%s sample_rate=%s",
            len(synthesized_segments),
            full_waveform.numel(),
            sample_rate,
        )

        return BilingualSpeechResult(
            waveform=full_waveform,
            sample_rate=sample_rate,
            segments=synthesized_segments,
            model_name_or_path=self.model_name_or_path,
        )

    def synthesize_segment(self, request: TTSSynthesisRequest) -> SynthesizedSegment:
        if not request.text.strip():
            raise ValueError("TTSSynthesisRequest.text must not be empty")

        conversation = self._build_conversation(request.text)

        try:
            model_inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = self._move_to_input_device(model_inputs)
            generated_text_ids, audio = self._generate_audio(
                model_inputs=model_inputs,
                speaker_name=request.voice.speaker_name,
            )
            sample_rate = int(
                getattr(
                    self.model.config, "audio_sampling_rate", request.voice.sample_rate
                )
            )
        except Exception as exc:
            self.logger.exception(
                "Qwen Omni TTS failed for segment %s voice=%s",
                request.segment_index,
                request.voice.voice_id,
            )
            raise RuntimeError("Qwen2.5-Omni TTS synthesis failed") from exc

        waveform = self._normalize_waveform(audio)

        self.logger.debug(
            "segment=%s voice=%s text_chars=%s generated_text_tokens=%s audio_samples=%s",
            request.segment_index,
            request.voice.voice_id,
            len(request.text),
            0 if generated_text_ids is None else generated_text_ids.numel(),
            waveform.numel(),
        )

        return SynthesizedSegment(
            text=request.text,
            voice=request.voice,
            waveform=waveform,
            sample_rate=sample_rate,
            segment_index=request.segment_index,
            start_char=request.start_char,
            end_char=request.end_char,
        )

    def _load_components(self) -> tuple[Any, Any]:
        try:
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            self.logger.exception(
                "transformers with Qwen2.5-Omni support is not installed"
            )
            raise RuntimeError(
                "transformers>=4.57 is required to use Qwen2.5-Omni TTS."
            ) from exc

        try:
            processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )

            load_kwargs = {
                "local_files_only": self.local_files_only,
                "torch_dtype": self.torch_dtype,
                "enable_audio_output": True,
            }
            if self.device.startswith("cuda"):
                load_kwargs["device_map"] = "auto"

            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                **load_kwargs,
            )

            if not self.device.startswith("cuda"):
                model = model.to(self.device)
            model.eval()
            return processor, model
        except Exception as exc:
            self.logger.exception(
                "Failed to load Qwen2.5-Omni model from %s", self.model_name_or_path
            )
            raise RuntimeError(
                "Failed to load Qwen2.5-Omni TTS. Ensure the model is available locally "
                "or authenticated through Hugging Face for offline download."
            ) from exc

    def _build_conversation(self, text: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": QWEN_OMNI_AUDIO_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Speak exactly the following text in a natural conversational voice. "
                            "Do not translate it, do not paraphrase it, and do not add extra words.\n"
                            f"{text}"
                        ),
                    }
                ],
            },
        ]

    def _generate_audio(
        self,
        model_inputs: Any,
        speaker_name: Optional[str],
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        generate_kwargs = {
            **dict(model_inputs),
            "return_audio": True,
            "thinker_do_sample": False,
            "talker_do_sample": False,
        }

        if speaker_name:
            try:
                output = self.model.generate(**generate_kwargs, speaker=speaker_name)
            except TypeError:
                output = self.model.generate(**generate_kwargs, spk=speaker_name)
        else:
            output = self.model.generate(**generate_kwargs)

        if isinstance(output, tuple) and len(output) == 2:
            return output[0], output[1]
        if isinstance(output, tuple) and len(output) > 2:
            return output[0], output[1]

        raise RuntimeError("Unexpected Qwen2.5-Omni generate output format")

    def _move_to_input_device(self, model_inputs: Any) -> Any:
        input_device = self._get_input_device()

        if hasattr(model_inputs, "to"):
            return model_inputs.to(input_device)

        if isinstance(model_inputs, dict):
            return {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }

        return model_inputs

    def _get_input_device(self) -> torch.device:
        thinker = getattr(self.model, "thinker", None)
        if thinker is not None and hasattr(thinker, "device"):
            return torch.device(thinker.device)

        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            return torch.device(model_device)

        return torch.device(self.device)

    def _normalize_waveform(self, audio: Any) -> torch.Tensor:
        waveform = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
        if waveform.ndim == 2:
            waveform = waveform.reshape(-1)
        elif waveform.ndim > 2:
            waveform = waveform.reshape(-1)
        return waveform.contiguous()

    def _concatenate_segments(
        self,
        segments: list[SynthesizedSegment],
        sample_rate: int,
    ) -> torch.Tensor:
        if not segments:
            return torch.zeros(0, dtype=torch.float32)

        pause_samples = int(sample_rate * self.pause_duration_ms / 1000.0)
        pause = torch.zeros(pause_samples, dtype=torch.float32)

        pieces: list[torch.Tensor] = []
        for index, segment in enumerate(segments):
            pieces.append(segment.waveform)
            if index < len(segments) - 1 and pause_samples > 0:
                pieces.append(pause)

        return torch.cat(pieces, dim=0)
