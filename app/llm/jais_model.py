from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Optional

import torch

from app.llm.prompt_builder import CodeSwitchPromptBuilder, ResponsePromptInput
from app.utils.logger import get_logger


DEFAULT_JAIS_MODEL = "mlconvexai/jais-13b-chat_bitsandbytes_4bit"
SUPPORTED_JAIS_MODELS = (
    "mlconvexai/jais-13b-chat_bitsandbytes_4bit",
    "models/mlconvexai/jais-13b-chat_bitsandbytes_4bit",
)


@dataclass
class JaisResponse:
    prompt: str
    response_text: str
    model_name_or_path: str


@dataclass
class JaisStreamChunk:
    text: str
    chunk_index: int


class JaisResponseGenerator:
    """Jais-13B-chat 4-bit (bitsandbytes) response generator for bilingual Arabic-English dialogue."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_JAIS_MODEL,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        local_files_only: bool = False,
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

    def generate_response(self, prompt_input: ResponsePromptInput) -> JaisResponse:
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
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            self.logger.exception("Jais generation failed for prompt: %s", prompt[:50])
            raise RuntimeError("Jais response generation failed") from exc

        self.logger.debug(
            "prompt_chars=%s response_chars=%s",
            len(prompt),
            len(response_text),
        )

        return JaisResponse(
            prompt=prompt,
            response_text=response_text,
            model_name_or_path=self.model_name_or_path,
        )

    def stream_response(
        self,
        prompt_input: ResponsePromptInput,
    ) -> tuple[str, list[JaisStreamChunk]]:
        """Stream response text chunks from Jais and return final text with chunks."""
        prompt = self.prompt_builder.build(prompt_input)

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            self.logger.exception("transformers is not installed")
            raise RuntimeError(
                "transformers is required to stream responses from the Jais generator."
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
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                except Exception as exc:
                    generation_error["error"] = exc

            generation_thread = Thread(target=_run_generation, daemon=True)
            generation_thread.start()

            chunk_index = 0
            chunks: list[JaisStreamChunk] = []
            response_parts: list[str] = []
            for text_chunk in streamer:
                if not text_chunk:
                    continue

                chunk_index += 1
                chunks.append(JaisStreamChunk(text=text_chunk, chunk_index=chunk_index))
                response_parts.append(text_chunk)

            generation_thread.join()
            if "error" in generation_error:
                raise generation_error["error"]

            response_text = "".join(response_parts).strip()
        except Exception as exc:
            self.logger.exception("Jais streaming generation failed")
            raise RuntimeError("Jais streaming response generation failed") from exc

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
                "transformers is required to load the Jais response generator."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
                trust_remote_code=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = {
                "local_files_only": self.local_files_only,
                "trust_remote_code": True,
                "device_map": "auto" if self.device.startswith("cuda") else None,
                "low_cpu_mem_usage": True,
            }

            if not load_kwargs["device_map"]:
                load_kwargs.pop("device_map")

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
                "Failed to load Jais model from %s", self.model_name_or_path
            )
            raise RuntimeError(
                "Failed to load Jais model. Ensure the model is available locally or authenticated on HuggingFace."
            ) from exc

    def _render_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful bilingual Arabic-English assistant."},
            {"role": "user", "content": prompt},
        ]

        chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(chat_template):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"### Instruction:\n{prompt}\n\n### Response:\n"

    def _move_to_device(self, model_inputs: Any) -> Any:
        if hasattr(model_inputs, "to"):
            return model_inputs.to(self.device)

        if isinstance(model_inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in model_inputs.items()
            }

        return model_inputs
