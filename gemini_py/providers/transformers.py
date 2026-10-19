"""Transformers provider — runs HuggingFace models locally.

Heavy: requires `pip install transformers torch`. Lazy-loaded on
construction. Default model is small (SmolLM2-360M) so it runs on
CPU. Override with `--provider-model <hf-repo>` to use a bigger one.

Supports function-calling via the tokenizer's chat template `tools=`
parameter (for models that have a tool-aware template — Qwen2.5,
Llama-3.x, Mistral-Nemo, Hermes, etc.). Falls back to injecting the
tool list as a system message if the template doesn't accept `tools=`.
Parses tool-call markers (`<|tool_call|>...<|/tool_call|>`,
`【{...}】`, fenced ```tool``` blocks) out of the model's output text
and converts them into _FunctionCallPart objects.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import Provider
from ._converters import contents_to_chat_messages, extract_user_prompt
from .._response import (
    _StreamedResponse,
    _TextPart,
    _UsageMetadata,
    text_response,
    text_with_tool_calls_response,
)


DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"


# Tool-call markers recognized in the model's output text.
_TOOL_CALL_PATTERNS = [
    # Hermes / Llama-3 style: <|tool_call|>{...}<|/tool_call|>
    re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", re.DOTALL),
    # Qwen2.5 style: 【{...}】
    re.compile(r"【\s*(\{.*?\})\s*】", re.DOTALL),
    # Mistral style: [TOOL_CALLS]  {...}
    re.compile(r"\[TOOL_CALLS\]\s*(\{.*?\})", re.DOTALL),
    # Fenced code block: ```tool\n{...}\n``` or ```json\n{...}\n```
    re.compile(r"```(?:tool|json|function_call)\s*\n?(\{.*?\})\n?```", re.DOTALL),
]


class TransformersProvider(Provider):
    """Local HuggingFace transformers provider with function-calling support."""

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Detect if the chat template accepts a `tools=` kwarg.
        self._template_supports_tools = self._check_template_supports_tools()

    # --- Provider protocol ------------------------------------------------

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        # Extract tools from config (Google function_declarations format).
        google_tools = config.get("tools") or []
        hf_tools = self._tools_to_hf_format(google_tools)
        prompt = self._build_prompt(contents, hf_tools)
        text = self._generate(prompt, config)
        tool_calls, remaining_text = _parse_tool_calls(text)
        if tool_calls:
            return text_with_tool_calls_response(remaining_text, tool_calls)
        return _StreamedResponse(
            parts=[_TextPart(text)],
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
        # We generate the full response first, then parse for tool-call
        # markers — true incremental streaming would prevent us from
        # reliably detecting tool calls mid-stream. Emit one chunk.
        google_tools = config.get("tools") or []
        hf_tools = self._tools_to_hf_format(google_tools)
        prompt = self._build_prompt(contents, hf_tools)
        text = self._generate(prompt, config)
        tool_calls, remaining_text = _parse_tool_calls(text)
        if tool_calls:
            yield text_with_tool_calls_response(remaining_text, tool_calls)
        else:
            yield _StreamedResponse(
                parts=[_TextPart(text)],
                text=text,
                usage_metadata=None,
                raw=None,
            )

    def list_models(self) -> List[str]:
        return [
            "HuggingFaceTB/SmolLM2-360M",
            "HuggingFaceTB/SmolLM2-1.7B",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-Nemo-Instruct-2407",
            "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "microsoft/Phi-3-mini-4k-instruct",
        ]

    def supports_tools(self) -> bool:
        # Yes — we parse tool calls out of the model's output, so the
        # agent loop can dispatch + execute them.
        return True

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

    def _check_template_supports_tools(self) -> bool:
        """Heuristic: try apply_chat_template with a dummy tools list."""
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "test_fn",
                        "description": "test",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }],
                add_generation_prompt=True,
                tokenize=False,
            )
            return True
        except Exception:
            return False

    def _tools_to_hf_format(self, google_tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert Google function_declarations to HuggingFace tools format."""
        out: List[Dict[str, Any]] = []
        for tool_group in google_tools or []:
            if not isinstance(tool_group, dict):
                continue
            decls = tool_group.get("function_declarations") or []
            for d in decls:
                if not isinstance(d, dict):
                    continue
                name = d.get("name") or ""
                if not name:
                    continue
                out.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": d.get("description", ""),
                        "parameters": d.get("parameters") or {"type": "object", "properties": {}},
                    },
                })
        return out

    def _build_prompt(
        self,
        contents: List[Dict[str, Any]],
        hf_tools: List[Dict[str, Any]],
    ) -> str:
        """Build the model input prompt via the tokenizer's chat template."""
        messages = contents_to_chat_messages(contents)

        # Try the chat template with tools first.
        if hf_tools and self._template_supports_tools:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=hf_tools,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as e:
                sys.stderr.write(
                    f"[gemini-py/transformers] apply_chat_template(tools=...) failed: {e}\n"
                )

        # Fallback: prepend a system message describing the tools.
        if hf_tools:
            tools_str = json.dumps(hf_tools, indent=2, ensure_ascii=False)
            sys_msg = {
                "role": "system",
                "content": (
                    "You can call tools to perform actions. Available tools:\n\n"
                    f"{tools_str}\n\n"
                    "When you want to call a tool, output ONLY the tool call "
                    "in this exact format (no extra text):\n"
                    "```tool\n"
                    '{"name": "<tool_name>", "arguments": {"<arg1>": "<value1>"}}\n'
                    "```\n"
                    "Wait for the tool result before responding further."
                ),
            }
            messages = [sys_msg] + messages

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as e:
            sys.stderr.write(
                f"[gemini-py/transformers] apply_chat_template failed: {e}; "
                f"falling back to raw text concat.\n"
            )
            return extract_user_prompt(contents)

    def _generate(self, prompt: str, config: Dict[str, Any]) -> str:
        """Generate text with full resilience against OOM / crashes / overflows.

        Handles:
          - Input too long (truncates to model's max_position_embeddings)
          - CUDA OOM (falls back to CPU automatically)
          - Generation crashes (returns error string, doesn't raise —
            so the agent loop can keep going with what we got)
          - Decode failures (returns error string)
          - Max tokens cap respected (generation stops, partial decode)

        Returns the generated text, OR an error string starting with
        '[generation failed: ...]' / '[tokenize error: ...]' on failures.

        Note: we deliberately DON'T import torch at the top of this
        method (only inside _call_generate / _move_to_cpu) so tests can
        mock those methods without needing torch installed.
        """
        max_new = int(config.get("max_output_tokens", 8192) or 8192)
        # Sensible default cap for local inference. We don't override the
        # user's explicit choice if they want more (or less).
        if max_new > 8192:
            max_new = 8192
        temperature = float(config.get("temperature", 1.0) or 1.0)
        top_p = float(config.get("top_p", 0.95) or 0.95)

        # 1. Tokenize with truncation (input too long → truncate).
        try:
            inputs = self._safe_tokenize(prompt)
        except Exception as e:
            sys.stderr.write(
                f"[gemini-py/transformers] tokenize failed: {e}\n"
            )
            return f"[tokenize error: {e}]"

        # 2. Generate with CUDA OOM fallback. We catch generic Exception
        # and check the message for OOM/CUDA signals (so we don't need to
        # import torch here just to reference torch.cuda.OutOfMemoryError).
        outputs = None
        last_error: Optional[Exception] = None
        try:
            outputs = self._call_generate(inputs, max_new, temperature, top_p)
        except Exception as e:
            err_msg = str(e).lower()
            is_oom = (
                "out of memory" in err_msg
                or "cuda" in err_msg
                or "oom" in err_msg
            )
            if is_oom:
                sys.stderr.write(
                    "[gemini-py/transformers] CUDA OOM, falling back to CPU\n"
                )
                try:
                    self._move_to_cpu()
                    inputs = self._safe_tokenize(prompt)  # re-tokenize for CPU
                    outputs = self._call_generate(
                        inputs, max_new, temperature, top_p
                    )
                except Exception as e2:
                    last_error = e2
            else:
                last_error = e

        if outputs is None:
            # Generation failed entirely — return an error string so the
            # agent loop can keep going (it'll see this as the model's
            # response and decide what to do next).
            sys.stderr.write(
                f"[gemini-py/transformers] generation failed: {last_error}\n"
            )
            return f"[generation failed: {last_error}]"

        # 3. Decode what we got — even if generation was cut short
        # (max_new_tokens reached, EOS, or partial).
        try:
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[0][input_len:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if not text.strip():
                # Empty output — model may have hit a limit or refused.
                return "[no output generated — model may have hit a limit or refused]"
            return text
        except Exception as e:
            sys.stderr.write(
                f"[gemini-py/transformers] decode failed: {e}\n"
            )
            return f"[decode error: {e}]"

    # --- Helpers for resilience -------------------------------------------

    def _get_max_input_tokens(self) -> int:
        """Get the model's max input length from config."""
        try:
            n = getattr(self.model.config, "max_position_embeddings", None)
            if n and isinstance(n, int) and n > 0:
                return n
        except Exception:
            pass
        # Sensible default — most chat models support at least 4096.
        return 4096

    def _safe_tokenize(self, prompt: str):
        """Tokenize, truncating input if it exceeds the model's max length."""
        max_len = self._get_max_input_tokens()
        # Reserve room for output tokens (don't fill the entire context
        # with input — leave at least 1024 for generation).
        max_input = max(256, max_len - 1024)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )
        return inputs.to(self.device)

    def _call_generate(self, inputs, max_new: int, temperature: float, top_p: float):
        """Call model.generate() — may raise on OOM or other failures."""
        import torch  # type: ignore
        with torch.no_grad():
            return self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

    def _move_to_cpu(self) -> None:
        """Move the model to CPU (for OOM recovery)."""
        try:
            import torch  # type: ignore
            self.device = "cpu"
            self.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            sys.stderr.write(
                f"[gemini-py/transformers] CPU fallback failed: {e}\n"
            )


# --- Tool-call parser ------------------------------------------------------


def _parse_tool_calls(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """Parse tool-call markers out of the model's text output.

    Returns (tool_calls, remaining_text) where:
        tool_calls: List of {"name": str, "args": dict, "id": Optional[str]}
        remaining_text: the input text with all tool-call markers removed.

    Recognizes:
        - ``<|tool_call|>{...}<|/tool_call|>``  (Hermes / Llama-3)
        - ``\\u3010{...}\\u3011``  (Qwen2.5 style, written as actual chars)
        - ``[TOOL_CALLS] {...}``  (Mistral style)
        - ``\\`\\`\\`tool\\n{...}\\n\\`\\`\\```  (fenced code block)
        - bare ``{"name": "...", "arguments": {...}}`` JSON
    """
    if not text:
        return [], text

    tool_calls: List[Dict[str, Any]] = []
    remaining = text

    for pattern in _TOOL_CALL_PATTERNS:
        for match in pattern.finditer(remaining):
            json_str = match.group(1).strip()
            tool_calls.extend(_parse_tool_call_json(json_str))
        # Remove matched regions from the remaining text.
        remaining = pattern.sub("", remaining)

    # Bare JSON: only if no markers matched, to avoid false positives.
    if not tool_calls:
        bare_pattern = re.compile(
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|args)"\s*:\s*\{[^}]*\}\s*\}',
            re.DOTALL,
        )
        for match in bare_pattern.finditer(text):
            tool_calls.extend(_parse_tool_call_json(match.group(0)))
        if tool_calls:
            remaining = bare_pattern.sub("", text)

    # Clean up extra whitespace.
    remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
    return tool_calls, remaining


def _parse_tool_call_json(json_str: str) -> List[Dict[str, Any]]:
    """Parse one or more tool calls from a JSON string.

    Accepts:
        {"name": "x", "arguments": {...}}
        {"name": "x", "args": {...}}
        {"function": "x", "arguments": {...}}
        [ {...}, {...} ]  (list of calls)
        {...}\\n{...}  (concatenated JSON objects)
    """
    out: List[Dict[str, Any]] = []
    # Single object.
    try:
        obj = json.loads(json_str)
        out.extend(_normalize_tool_call_obj(obj))
        return out
    except json.JSONDecodeError:
        pass

    # List of objects — handle whitespace between consecutive JSON objects.
    try:
        # Normalize "}{ " (with optional whitespace) → "},{"
        normalized = re.sub(r"\}\s*\{", "},{", json_str)
        obj = json.loads(f"[{normalized}]")
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.extend(_normalize_tool_call_obj(item))
            return out
    except json.JSONDecodeError:
        pass

    # Concatenated JSON objects.
    for m in re.finditer(r"\{[^{}]*\}", json_str):
        try:
            obj = json.loads(m.group(0))
            out.extend(_normalize_tool_call_obj(obj))
        except json.JSONDecodeError:
            continue

    return out


def _normalize_tool_call_obj(obj: Any) -> List[Dict[str, Any]]:
    """Normalize a parsed JSON object into our tool-call dict shape."""
    if not isinstance(obj, dict):
        return []
    name = obj.get("name") or obj.get("function") or ""
    if not name:
        return []
    args = obj.get("arguments") or obj.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    call_id = obj.get("id")
    return [{"name": name, "args": args, "id": call_id}]
