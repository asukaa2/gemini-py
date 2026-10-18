"""GeminiClient — the main SDK client.

Wraps `google.genai.Client` and provides high-level methods:
    - generate_text(prompt)            -> GenerateResult
    - chat(messages)                   -> GenerateResult (multi-turn)
    - stream(prompt)                   -> iterator of text chunks
    - run_headless(prompt, ...)        -> headless one-shot (mirrors `gemini -p`)
    - run_agent_turn(prompt, tools)    -> agentic flow with tool calls
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from .auth import AuthConfig
from .config import GeminiConfig
from .models import GenerateResult, Message, MessageRole, UsageStats
from .prompts import build_system_prompt
from .sessions import SessionStore
from .tools import (
    Tool,
    build_default_tools,
    execute_tool_call,
    tool_declarations,
)


class GeminiError(Exception):
    """Raised when the Gemini API returns an error or input is invalid."""


class GeminiClient:
    """High-level client for Google Gemini.

    Args:
        auth: Authentication configuration. Defaults to `AuthConfig.from_env()`.
        config: Generation config. Defaults to a sensible `GeminiConfig()`.

    Example:
        >>> from gemini_py import GeminiClient
        >>> client = GeminiClient()
        >>> r = client.generate_text("Say hello.")
        >>> print(r.text)
        'Hello! How can I help you today?'
    """

    def __init__(
        self,
        auth: Optional[AuthConfig] = None,
        config: Optional[GeminiConfig] = None,
    ) -> None:
        self.auth = auth or AuthConfig.from_env()
        self.config = config or GeminiConfig()
        self._client = self._build_underlying_client()

    # --- Construction -----------------------------------------------------

    def _build_underlying_client(self):
        """Build the underlying `google.genai.Client` from the auth config."""
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise GeminiError(
                "google-genai is not installed. Run `pip install google-genai`."
            ) from exc

        try:
            kwargs = self.auth.to_client_kwargs()
        except ValueError as e:
            raise GeminiError(
                f"Authentication not configured: {e}\n"
                f"Set GEMINI_API_KEY (from https://aistudio.google.com/app/apikey) "
                f"or use Vertex AI (set GOOGLE_CLOUD_PROJECT + ADC)."
            ) from e
        try:
            return genai.Client(**kwargs)
        except Exception as e:
            raise GeminiError(f"Failed to initialize google.genai.Client: {e}") from e

    # --- Simple generation ------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        config: Optional[GeminiConfig] = None,
        system: Optional[str] = None,
    ) -> GenerateResult:
        """Generate a one-shot text response for `prompt`.

        Args:
            prompt: The user's prompt text.
            config: Optional override config.
            system: Optional system instruction (overrides config).

        Returns:
            GenerateResult with the model's text, usage, and metadata.
        """
        cfg = config or self.config
        if system:
            cfg = GeminiConfig(**{**cfg.to_dict(), "system_instruction": system})

        gen_cfg = cfg.to_generate_content_config()
        model = cfg.resolve_model()

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_cfg,
            )
        except Exception as e:
            raise GeminiError(f"generate_content failed: {e}") from e

        return self._build_result(response, cfg)

    def chat(
        self,
        messages: List[Message],
        *,
        config: Optional[GeminiConfig] = None,
    ) -> GenerateResult:
        """Generate a response given a multi-turn message history.

        Args:
            messages: The conversation so far (as Message objects).
            config: Optional override config.

        Returns:
            GenerateResult for the model's reply.
        """
        cfg = config or self.config
        gen_cfg = cfg.to_generate_content_config()
        model = cfg.resolve_model()

        contents = [m.to_dict() for m in messages]
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_cfg,
            )
        except Exception as e:
            raise GeminiError(f"chat generate failed: {e}") from e

        return self._build_result(response, cfg)

    # --- Streaming --------------------------------------------------------

    def stream(
        self,
        prompt: str,
        *,
        config: Optional[GeminiConfig] = None,
    ) -> Iterator[str]:
        """Stream text chunks for `prompt`.

        Yields incremental text deltas. Useful for interactive REPLs.
        """
        cfg = config or self.config
        gen_cfg = cfg.to_generate_content_config()
        model = cfg.resolve_model()
        try:
            stream = self._client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=gen_cfg,
            )
            for chunk in stream:
                text = getattr(chunk, "text", "") or ""
                if text:
                    yield text
        except Exception as e:
            raise GeminiError(f"streaming failed: {e}") from e

    # --- Agentic flow -----------------------------------------------------

    def run_agent_turn(
        self,
        prompt: str,
        *,
        tools: Optional[List[Tool]] = None,
        config: Optional[GeminiConfig] = None,
        max_iterations: int = 10,
        extra_context: Optional[str] = None,
        on_event: Optional[Any] = None,
    ) -> GenerateResult:
        """Run a single agent turn with tool-calling.

        The model can call tools; we execute them and feed back the results
        until the model produces a final text response or `max_iterations`
        is reached.

        Args:
            prompt: User's prompt.
            tools: Tools to expose (default: filesystem + shell).
            config: Override config.
            max_iterations: Max tool-call round-trips.
            extra_context: Extra context for the system prompt.
            on_event: Optional callback(event: dict) for each tool call.

        Returns:
            GenerateResult with the final answer + tool history.
        """
        cfg = config or self.config
        tools = tools if tools is not None else build_default_tools(
            enable_shell=cfg.enable_shell_tool,
            enable_filesystem=cfg.enable_filesystem_tools,
            enable_web=cfg.enable_web_search or cfg.enable_web_fetch,
        )
        declarations = tool_declarations(tools)

        # Build the system instruction
        system = cfg.system_instruction or build_system_prompt(
            enable_tools=bool(tools),
            extra_context=extra_context,
        )
        gen_cfg = {
            **cfg.to_generate_content_config(),
            "system_instruction": system,
            "tools": declarations,
        }
        model = cfg.resolve_model()

        contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
        tool_results: List[Dict[str, Any]] = []

        for _ in range(max_iterations):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                )
            except Exception as e:
                raise GeminiError(f"agent iteration failed: {e}") from e

            # Extract any function calls
            candidates = getattr(response, "candidates", []) or []
            function_calls = []
            text_parts: List[str] = []

            for cand in candidates:
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                parts = getattr(content, "parts", []) or []
                for part in parts:
                    # Text part
                    t = getattr(part, "text", None)
                    if t:
                        text_parts.append(t)
                    # Function call part
                    fc = getattr(part, "function_call", None)
                    if fc:
                        function_calls.append(
                            {
                                "name": getattr(fc, "name", ""),
                                "args": dict(getattr(fc, "args", {}) or {}),
                            }
                        )

            if not function_calls:
                # No tool calls — this is the final answer.
                usage = UsageStats.from_metadata(getattr(response, "usage_metadata", None))
                return GenerateResult(
                    text="\n".join(text_parts),
                    usage=usage,
                    model=model,
                    finish_reason=str(getattr(candidates[0], "finish_reason", "") if candidates else ""),
                    tool_results=tool_results,
                    raw=response,
                )

            # Append the model's response (with calls) to the conversation.
            contents.append(
                {
                    "role": "model",
                    "parts": [{"function_call": {"name": fc["name"], "args": fc["args"]}} for fc in function_calls],
                }
            )

            # Execute each function call and append responses.
            response_parts = []
            for fc in function_calls:
                if on_event:
                    on_event({"type": "tool_call", "name": fc["name"], "args": fc["args"]})
                result = execute_tool_call(tools, fc)
                response_parts.append(result)
                tool_results.append(
                    {
                        "name": fc["name"],
                        "args": fc["args"],
                        "result": result["function_response"]["response"]["result"],
                    }
                )
                if on_event:
                    on_event(
                        {
                            "type": "tool_result",
                            "name": fc["name"],
                            "result": result["function_response"]["response"]["result"][:500],
                        }
                    )

            contents.append({"role": "user", "parts": response_parts})

        # Exceeded iterations
        return GenerateResult(
            text="(max tool-call iterations reached)",
            usage=UsageStats(),
            model=model,
            tool_results=tool_results,
        )

    # --- Headless one-shot ------------------------------------------------

    def run_headless(
        self,
        prompt: str,
        *,
        config: Optional[GeminiConfig] = None,
        tools: Optional[List[Tool]] = None,
        stdin: Optional[str] = None,
        output_format: str = "text",
        stream_events: bool = False,
        on_event: Optional[Any] = None,
    ) -> Union[GenerateResult, Iterator[Dict[str, Any]]]:
        """Run a headless (non-interactive) request, mirroring `gemini -p`.

        Args:
            prompt: The user prompt.
            config: Override config.
            tools: Tools to enable. None means default tools.
            stdin: Extra stdin content appended to the prompt.
            output_format: 'text' (default), 'json', or 'stream-json'.
            stream_events: If True, yield events instead of returning a result.
            on_event: Optional callback for events.
        """
        cfg = config or self.config

        # Combine stdin + prompt like gemini-cli.
        full_prompt = prompt
        if stdin:
            full_prompt = (stdin + "\n\n" + prompt) if prompt else stdin

        if stream_events:
            return self._stream_events(
                full_prompt,
                cfg=cfg,
                tools=tools,
                on_event=on_event,
            )

        result = self.run_agent_turn(
            full_prompt,
            tools=tools,
            config=cfg,
            on_event=on_event,
        )
        return result

    def _stream_events(
        self,
        prompt: str,
        *,
        cfg: GeminiConfig,
        tools: Optional[List[Tool]],
        on_event,
    ) -> Iterator[Dict[str, Any]]:
        """Yield JSONL-style events mirroring gemini-cli stream-json output."""
        session_id = None
        # Emit init
        init_event = {"type": "init", "model": cfg.resolve_model(), "session_id": session_id}
        yield init_event
        if on_event:
            on_event(init_event)

        # Emit user message
        user_event = {"type": "message", "role": "user", "content": prompt}
        yield user_event
        if on_event:
            on_event(user_event)

        def _local_event(ev):
            yield ev
            if on_event:
                on_event(ev)

        # Run the agent and emit tool events, then a final result.
        result = self.run_agent_turn(
            prompt,
            tools=tools,
            config=cfg,
            extra_context=None,
            on_event=lambda ev: None,  # we'll re-emit below
        )

        for tr in result.tool_results:
            yield {"type": "tool_use", "name": tr["name"], "args": tr["args"]}
            yield {"type": "tool_result", "name": tr["name"], "result": tr["result"]}

        # Final result
        yield {
            "type": "result",
            "response": result.text,
            "stats": result.usage.to_dict(),
            "model": result.model,
        }

    # --- Internals --------------------------------------------------------

    def _build_result(self, response: Any, cfg: GeminiConfig) -> GenerateResult:
        """Convert a google.genai response into our GenerateResult."""
        text = getattr(response, "text", "") or ""
        usage = UsageStats.from_metadata(getattr(response, "usage_metadata", None))
        candidates = getattr(response, "candidates", []) or []
        finish_reason = ""
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
        return GenerateResult(
            text=text,
            usage=usage,
            model=cfg.resolve_model(),
            finish_reason=finish_reason,
            raw=response,
        )


# --- Convenience top-level function -----------------------------------------


def run_headless(prompt: str, **kwargs) -> GenerateResult:
    """Module-level convenience for `GeminiClient().run_headless`.

    Useful for one-off scripts:

        from gemini_py import run_headless
        print(run_headless("Summarize this code.").text)
    """
    client = GeminiClient()
    return client.run_headless(prompt, **kwargs)
