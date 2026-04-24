from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from app.llm.prompt_builder import CodeSwitchPromptBuilder, ResponsePromptInput
from app.utils.logger import get_logger


DEFAULT_GEMMA_MODEL = "google/gemma-2b-it"
SUPPORTED_GEMMA_MODELS = ("google/gemma-2b-it", "google/gemma-7b-it")


@dataclass
class GemmaResponse:
    prompt: str
    response_text: str
    model_name_or_path: str


class GemmaResponseGenerator:
    """Gemma-based response generator for code-switched bilingual dialogue."""

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_GEMMA_MODEL,
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

    def generate_response(self, prompt_input: ResponsePromptInput) -> GemmaResponse:
        prompt = self.prompt_builder.build(prompt_input)

        try:
            rendered_prompt = self._render_prompt(prompt)
            model_inputs = self.tokenizer(
                rendered_prompt,
                return_tensors="pt",
            )
            model_inputs = self._move_to_device(model_inputs)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.temperature > 0.0,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = model_inputs["input_ids"].shape[-1]
            generated_only = generated_ids[:, prompt_length:]
            response_text = self.tokenizer.decode(
                generated_only[0],
                skip_special_tokens=True,
            ).strip()
        except Exception as exc:
            self.logger.exception("Gemma generation failed")
            raise RuntimeError("Gemma response generation failed") from exc

        self.logger.debug(
            "model=%s prompt_chars=%s response_chars=%s",
            self.model_name_or_path,
            len(prompt),
            len(response_text),
        )

        return GemmaResponse(
            prompt=prompt,
            response_text=response_text,
            model_name_or_path=self.model_name_or_path,
        )

    def _load_components(self) -> tuple[Any, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            self.logger.exception("transformers is not installed")
            raise RuntimeError(
                "transformers is required to load the Gemma response generatorgoogle/gemma-2b-it."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                local_files_only=self.local_files_only,
            )

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
            self.logger.exception("Failed to load Gemma model from %s", self.model_name_or_path)
            raise RuntimeError(
                "Failed to load Gemma. Ensure you accepted the Gemma license on Hugging Face "
                "and authenticated with `hf auth login`, or point to a downloaded local path."
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
