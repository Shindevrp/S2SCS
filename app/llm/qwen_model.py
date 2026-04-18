from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Optional

import torch

from app.llm.prompt_builder import CodeSwitchPromptBuilder, ResponsePromptInput
from app.utils.logger import get_logger


DEFAULT_QWEN_MODEL = "models/Qwen/Qwen2.5-7B-Instruct"
SUPPORTED_QWEN_MODELS = (
    "models/Qwen/Qwen2.5-7B-Instruct",
    "models/Qwen/Qwen2.5-1.5B-Instruct",
)


@dataclass
class QwenResponse:
    prompt: str
    response_text: str
    model_name_or_path: str


@dataclass
class QwenStreamChunk:
    text: str
    chunk_index: int


class QwenResponseGenerator:
    """Qwen-based response generator for code-switched bilingual dialogue."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_QWEN_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        local_files_only: bool = True,
        prompt_builder: Optional[CodeSwitchPromptBuilder] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.local_files_only = local_files_only
        self.prompt_builder = prompt_builder or CodeSwitchPromptBuilder()
        self.logger = get_logger(self.__class__.__name__)

        self.tokenizer = tokenizer
        self.model = model
        if self.tokenizer is None or self.model is None:
            self.tokenizer, self.model = self._load_components()

    def generate_response(self, prompt_input: ResponsePromptInput) -> QwenResponse:
        prompt = self.prompt_builder.build(prompt_input)

        try:
            rendered_prompt = self._render_prompt(prompt)
            inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
            inputs = self._move_to_device(inputs)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )

            response_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            self.logger.exception("Qwen generation failed for prompt: %s", prompt[:50])
            raise RuntimeError("Qwen response generation failed") from exc

        self.logger.debug(
            "prompt_chars=%s response_chars=%s",
            len(prompt),
            len(response_text),
        )

        return QwenResponse(
            prompt=prompt,
            response_text=response_text,
            model_name_or_path=self.model_name_or_path,
        )

    def stream_response(
        self,
        prompt_input: ResponsePromptInput,
    ) -> tuple[str, list[QwenStreamChunk]]:
        """Stream response text chunks from Qwen and return final text with chunks."""
        prompt = self.prompt_builder.build(prompt_input)

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            self.logger.exception("transformers is not installed")
            raise RuntimeError(
                "transformers is required to stream responses from the Qwen generator."
            ) from exc

        try:
            rendered_prompt = self._render_prompt(prompt)
            inputs = self.tokenizer(rendered_prompt, return_tensors="pt")
            inputs = self._move_to_device(inputs)

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_error: dict[str, Exception] = {}

            def _run_generation() -> None:
                try:
                    with torch.inference_mode():
                        self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            do_sample=True,
                            streamer=streamer,
                        )
                except Exception as exc:
                    generation_error["error"] = exc

            generation_thread = Thread(target=_run_generation, daemon=True)
            generation_thread.start()

            chunk_index = 0
            chunks: list[QwenStreamChunk] = []
            response_parts: list[str] = []
            for text_chunk in streamer:
                if not text_chunk:
                    continue

                chunk_index += 1
                chunks.append(QwenStreamChunk(text=text_chunk, chunk_index=chunk_index))
                response_parts.append(text_chunk)

            generation_thread.join()
            if "error" in generation_error:
                raise generation_error["error"]

            response_text = "".join(response_parts).strip()
        except Exception as exc:
            self.logger.exception("Qwen streaming generation failed")
            raise RuntimeError("Qwen streaming response generation failed") from exc

        self.logger.debug(
            "stream_prompt_chars=%s stream_chunks=%s response_chars=%s",
            len(prompt),
            len(chunks),
            len(response_text),
        )

        return response_text, chunks

    def _load_components(self) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self.logger.exception("transformers is not installed")
            raise RuntimeError(
                "transformers is required to load the Qwen response generator."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = {
                "local_files_only": self.local_files_only,
                "torch_dtype": self.torch_dtype,
            }
            if self.device.startswith("cuda"):
                load_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                **load_kwargs,
            )

            if not self.device.startswith("cuda"):
                model = model.to(self.device)
            model.eval()
            return tokenizer, model
        except Exception as exc:
            self.logger.exception(
                "Failed to load Qwen model from %s", self.model_name_or_path
            )
            raise RuntimeError(
                "Failed to load Qwen model. Ensure the model is available locally."
            ) from exc

    def _render_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(chat_template):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return prompt

    def _move_to_device(self, model_inputs: Any) -> Any:
        if hasattr(model_inputs, "to"):
            return model_inputs.to(self.device)

        if isinstance(model_inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }

        return model_inputs
