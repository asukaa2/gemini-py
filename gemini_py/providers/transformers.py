"""Transformers provider — runs HuggingFace models locally.

Heavy: requires `pip install transformers torch`. Lazy-loaded on
construction. Default model is small (SmolLM2-360M) so it runs on
CPU. Override with `--provider-model <hf-repo>` to use a bigger one.

Text-only — no function-calling, no multi-turn conversation
(stateless single-shot generation).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from . import Provider
from ._converters import extract_user_prompt
from .._response import _StreamedResponse, _UsageMetadata


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


class TransformersProvider(Provider):
    """Local HuggingFace transformers provider."""

    def __init__(self, *, model_name: Optional[str] = None) -> None:
        try:
            import torch  # noqa: F401  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch are not installed. Run "
                "`pip install transformers torch`."
            ) from e

        import torch  # type: ignore
        self.model_name = model_name or DEFAULT_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load tokenizer + model (this can take a while on first run).
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        # Some tokenizers don't have a pad token — set one.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        # Ignore `model` arg (we use the model loaded at construction).
        prompt = extract_user_prompt(contents)
        text = self._generate(prompt, config)
        return _StreamedResponse(
            parts=[],
            text=text,
            usage_metadata=None,
            raw=None,
            finish_reason="STOP",
        )

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        # Transformers supports streaming via TextIteratorStreamer, but to
        # keep deps simple, we just emit the full text as a single chunk.
        prompt = extract_user_prompt(contents)
        text = self._generate(prompt, config)
        yield _StreamedResponse(
            parts=[],
            text=text,
            usage_metadata=None,
            raw=None,
        )

    def list_models(self) -> List[str]:
        # Can't enumerate HF Hub without an API call; return some suggestions.
        return [
            "HuggingFaceTB/SmolLM2-360M",
            "HuggingFaceTB/SmolLM2-1.7B",
            "Qwen/Qwen2.5-0.5B",
            "Qwen/Qwen2.5-1.5B",
            "meta-llama/Llama-3.2-1B",
            "microsoft/Phi-3-mini-4k-instruct",
        ]

    def supports_tools(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "transformers"

    def close(self) -> None:
        try:
            import torch  # type: ignore
            del self.model
            del self.tokenizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --- Internal --------------------------------------------------------

    def _generate(self, prompt: str, config: Dict[str, Any]) -> str:
        import torch  # type: ignore
        max_new = config.get("max_output_tokens", 8192)
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 0.95)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Strip the prompt from the output.
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
