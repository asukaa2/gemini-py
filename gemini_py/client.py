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
    """High-level client for Gemini (and other providers).

    Args:
        auth: Authentication configuration (Gemini only). Defaults to
            `AuthConfig.from_env()` when `provider` is None or "gemini".
        config: Generation config. Defaults to a sensible `GeminiConfig()`.
        provider: A `Provider` instance, a provider name string
            ("gemini" | "openai" | "anthropic" | "ollama" | "transformers"),
            or None (default: build a GeminiProvider from `auth`).

    Example:
        >>> from gemini_py import GeminiClient
        >>> client = GeminiClient()
        >>> r = client.generate_text("Say hello.")
        >>> print(r.text)
        'Hello! How can I help you today?'

        >>> # Use Ollama instead:
        >>> from gemini_py.providers import make_provider
        >>> provider = make_provider("ollama", model="llama3.2")
        >>> client = GeminiClient(provider=provider)
        >>> print(client.generate_text("Hello").text)
    """

    def __init__(
        self,
        auth: Optional[AuthConfig] = None,
        config: Optional[GeminiConfig] = None,
        provider: Optional[Any] = None,
    ) -> None:
        self.config = config or GeminiConfig()
        # Resolve provider.
        if provider is None:
            # Default: Gemini with the user's auth.
            self.auth = auth or AuthConfig.from_env()
            from .providers.gemini import GeminiProvider
            self._provider = GeminiProvider(auth=self.auth)
        elif isinstance(provider, str):
            # Provider name — use factory.
            self.auth = auth
            from .providers import make_provider
            self._provider = make_provider(provider, auth=auth)
        else:
            # Provider instance.
            self.auth = auth
            self._provider = provider
        # Backward-compat: expose self._client as the provider (some
        # external code may have used `client._client.models.generate_content`).
        self._client = self._provider
        # Register this instance as the "last client" so the subagent tool
        # (which is built lazily inside run_agent_turn) can spawn a child turn
        # using the same auth + config + provider.
        _LAST_CLIENT_REF[0] = self

    # --- Construction (legacy, kept for backward compat) ------------------

    def _build_underlying_client(self):
        """Legacy: build a `google.genai.Client` directly.

        Kept for backward compat with code that called this method directly.
        New code should use `GeminiClient(provider=...)` instead.
        """
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
            response = self._provider.generate_content(
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
            response = self._provider.generate_content(
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
            stream = self._provider.generate_content_stream(
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
        # New integrations (default-on, can be opted out):
        enable_memory: bool = True,
        enable_skills: bool = True,
        enable_subagent: bool = True,
        mcp_client: Optional[Any] = None,
        skills_loader: Optional[Any] = None,
        # Live view: stream text chunks + tool calls + tool results in real
        # time. The on_event callback gets fed every event (text_chunk,
        # tool_call, tool_result, iteration, done, error, file_write).
        live: bool = False,
    ) -> GenerateResult:
        """Run a single agent turn with tool-calling.

        The model can call tools; we execute them and feed back the results
        until the model produces a final text response or `max_iterations`
        is reached.

        New integrations:
            - **Auto-memory**: hierarchical GEMINI.md files are loaded and
              appended to the system instruction (set `enable_memory=False`
              to skip).
            - **Skills**: `@skill_name` tokens in the prompt are expanded to
              the skill's content (set `enable_skills=False` to skip).
            - **Subagent**: the `delegate_subagent(skill, task)` tool is
              exposed so the model can spawn a focused subagent (set
              `enable_subagent=False` to skip).
            - **Live view**: when `live=True`, stream text chunks + tool
              calls + tool results to `on_event` in real-time. Use the
              `LiveView` class from `.live` for a rich stderr renderer.
            - **MCP**: pass an `mcp_client` (already `.start()`-ed) to
              expose its tools alongside the built-in ones.

        Args:
            prompt: User's prompt.
            tools: Tools to expose (default: filesystem + shell).
            config: Override config.
            max_iterations: Max tool-call round-trips.
            extra_context: Extra context for the system prompt.
            on_event: Optional callback(event: dict) for each tool call.
            enable_memory: Load hierarchical GEMINI.md context (default True).
            enable_skills: Expand @skill tokens + list skills in prompt.
            enable_subagent: Expose the delegate_subagent tool.
            mcp_client: An McpClient whose tools should be exposed.
            skills_loader: Optional pre-built SkillsLoader.

        Returns:
            GenerateResult with the final answer + tool history.
        """
        cfg = config or self.config
        tools = tools if tools is not None else build_default_tools(
            enable_shell=cfg.enable_shell_tool,
            enable_filesystem=cfg.enable_filesystem_tools,
            enable_web=cfg.enable_web_search or cfg.enable_web_fetch,
        )

        # Check provider's tool support. Non-Gemini providers don't yet
        # support function-calling, so the agent loop would loop forever
        # waiting for tool calls. Auto-disable tools + warn the user.
        provider_supports_tools = True
        try:
            provider_supports_tools = self._provider.supports_tools()
        except Exception:
            provider_supports_tools = True
        if not provider_supports_tools and tools:
            import sys as _sys
            _sys.stderr.write(
                f"[gemini-py] provider '{self._provider.name}' does not support "
                f"function-calling — disabling tools for this turn (text-only mode).\n"
            )
            tools = []

        # --- Skills loader (lazy init) ---------------------------------
        if skills_loader is None and enable_skills:
            try:
                from .skills import SkillsLoader
                skills_loader = SkillsLoader()
            except Exception:
                skills_loader = None

        # --- Expand @skill_name tokens in prompt -----------------------
        if enable_skills and skills_loader is not None:
            from .skills import expand_skills_in_prompt
            prompt = expand_skills_in_prompt(prompt, skills_loader)

        # --- Add the subagent tool -------------------------------------
        if enable_subagent:
            subagent_tool = _build_subagent_tool(skills_loader)
            if subagent_tool is not None:
                tools = list(tools) + [subagent_tool]

        # --- Declarations = built-in + MCP -----------------------------
        declarations = tool_declarations(tools)
        if mcp_client is not None:
            try:
                mcp_decls = mcp_client.list_tools()
                if mcp_decls:
                    # Combine under the same tools list (google-genai accepts
                    # a list of Tool dicts, each with function_declarations).
                    declarations = declarations + [{"function_declarations": mcp_decls}]
            except Exception as e:
                sys.stderr.write(
                    f"[gemini-py] MCP list_tools failed: {e}\n"
                )

        # --- Build system instruction with memory + skills summary -----
        if cfg.system_instruction:
            system = cfg.system_instruction
        else:
            system = build_system_prompt(enable_tools=bool(tools))
        # Append memory context
        if enable_memory:
            try:
                from .memory import load_memory
                mem = load_memory()
                if mem:
                    system = system + "\n\n# Project context (from GEMINI.md)\n" + mem
            except Exception as e:
                sys.stderr.write(f"[gemini-py] memory load failed: {e}\n")
        # Append extra context (e.g., stdin)
        if extra_context:
            system = system + "\n\n# Additional context\n" + extra_context
        # Append skills summary
        if enable_skills and skills_loader is not None:
            try:
                summary = skills_loader.render_summary()
                if summary:
                    system = system + "\n\n" + summary
            except Exception:
                pass

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
        # Track recent call fingerprints (name+sorted args) to detect when
        # the model is stuck in a loop re-issuing the same call.
        recent_calls: List[str] = []
        last_text_chunk: str = ""

        for iteration in range(max_iterations):
            if on_event:
                on_event({"type": "iteration", "n": iteration + 1, "max": max_iterations})

            if live:
                # Streaming path: emit text chunks in real time, then
                # collect function calls from the final accumulated state.
                response = self._stream_and_collect(
                    model=model,
                    contents=contents,
                    config=gen_cfg,
                    on_event=on_event,
                )
            else:
                try:
                    response = self._provider.generate_content(
                        model=model,
                        contents=contents,
                        config=gen_cfg,
                    )
                except Exception as e:
                    raise GeminiError(f"agent iteration failed: {e}") from e

            # Extract any function calls. We preserve the ORIGINAL Part
            # objects so we can pass them back to the API unchanged — the
            # Gemini API now requires `thought_signature` on function-call
            # parts (Oct 2026), and rebuilding parts from just name+args
            # drops that signature, causing HTTP 400 INVALID_ARGUMENT.
            candidates = getattr(response, "candidates", []) or []
            function_calls = []          # list of {"name", "args"} for exec
            model_parts: List[Any] = []  # original Part objects for the next round
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
                                # Preserve the call id so the API can link
                                # the function_response back to this call.
                                # Missing linkage causes the model to
                                # re-issue the same call every turn.
                                "id": getattr(fc, "id", None),
                            }
                        )
                        # Preserve the original Part (carries thought_signature).
                        model_parts.append(part)

            if not function_calls:
                # No tool calls — this is the final answer.
                usage = UsageStats.from_metadata(getattr(response, "usage_metadata", None))
                if on_event:
                    on_event(
                        {
                            "type": "done",
                            "finish_reason": str(
                                getattr(candidates[0], "finish_reason", "") if candidates else ""
                            ),
                            "n_tool_calls": len(tool_results),
                        }
                    )
                return GenerateResult(
                    text="\n".join(text_parts),
                    usage=usage,
                    model=model,
                    finish_reason=str(getattr(candidates[0], "finish_reason", "") if candidates else ""),
                    tool_results=tool_results,
                    raw=response,
                )

            # Loop detection: if the model makes the exact same call (by
            # name + sorted args) three times in a row, break out of the
            # loop and return whatever text we have so far. This is a
            # safety valve for cases where the model is stuck repeating
            # the same tool call (often because the result confuses it).
            current_fingerprints = [
                f"{fc['name']}:{json.dumps(fc.get('args') or {}, sort_keys=True)}"
                for fc in function_calls
            ]
            recent_calls.extend(current_fingerprints)
            # Keep only the last 6 fingerprints (3 iterations * up-to-2-calls).
            recent_calls = recent_calls[-6:]
            # If the same fingerprint appears 3+ times, treat as a loop.
            from collections import Counter
            counts = Counter(recent_calls)
            looped = [fp for fp, n in counts.items() if n >= 3]
            if looped:
                # If we have any text from the model, return it. Otherwise
                # produce a helpful diagnostic.
                summary_lines = []
                if text_parts:
                    summary_lines.extend(text_parts)
                summary_lines.append(
                    f"\n[Stopped: model repeated the same tool call "
                    f"{len(looped)} time(s) — likely stuck. Last calls: "
                    f"{', '.join(fp.split(':', 1)[0] for fp in looped)}.]"
                )
                usage = UsageStats.from_metadata(getattr(response, "usage_metadata", None))
                if on_event:
                    on_event(
                        {
                            "type": "done",
                            "finish_reason": "loop_detected",
                            "n_tool_calls": len(tool_results),
                        }
                    )
                return GenerateResult(
                    text="\n".join(summary_lines),
                    usage=usage,
                    model=model,
                    finish_reason="loop_detected",
                    tool_results=tool_results,
                    raw=response,
                )

            # Append the model's response (with calls) to the conversation.
            # Pass the original Part objects so thought_signature survives.
            contents.append(
                {
                    "role": "model",
                    "parts": model_parts,
                }
            )

            # Execute each function call and append responses.
            response_parts = []
            for fc in function_calls:
                # Special-case write_file: emit a file_write event so the
                # live view can render the code being written.
                if on_event and fc["name"] == "write_file":
                    on_event(
                        {
                            "type": "file_write",
                            "path": fc["args"].get("path", "?"),
                            "content": fc["args"].get("content", ""),
                        }
                    )
                if on_event:
                    on_event({"type": "tool_call", "name": fc["name"], "args": fc["args"]})
                result = self._dispatch_tool_call(
                    tools=tools,
                    fc=fc,
                    mcp_client=mcp_client,
                    skills_loader=skills_loader,
                )
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

        # Exceeded iterations without a final answer.
        # Provide a helpful diagnostic so the user knows what to tune.
        if on_event:
            on_event(
                {
                    "type": "error",
                    "message": f"max_iterations ({max_iterations}) reached without a final answer",
                }
            )
        return GenerateResult(
            text=(
                f"(agent hit the max-iterations limit of {max_iterations} "
                f"without producing a final answer). "
                f"Possible causes: (1) the model is stuck in a tool-call "
                f"loop — try `--max-iterations 25` or a more capable model "
                f"(`--model argon`); (2) a tool returned an error the model "
                f"can't recover from — check tool_results; (3) the task is "
                f"genuinely complex — raise `--max-iterations` or split it."
            ),
            usage=UsageStats(),
            model=model,
            finish_reason="max_iterations_reached",
            tool_results=tool_results,
        )

    def _stream_and_collect(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
        on_event: Optional[Any] = None,
    ) -> Any:
        """Stream a generate_content call and accumulate chunks.

        Emits each chunk's text via on_event as `text_chunk` events for
        live view. Returns a synthesized response object with all parts
        (text + function_call) accumulated from the chunks.

        We can't simply return the last chunk because text and
        function_calls may be split across multiple chunks.
        """
        try:
            stream = self._provider.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            raise GeminiError(f"streaming iteration failed: {e}") from e

        accumulated_text_parts: List[str] = []
        accumulated_parts: List[Any] = []  # original Part objects
        last_response: Any = None
        last_usage: Any = None

        try:
            for chunk in stream:
                last_response = chunk
                # Capture usage_metadata (typically on the last chunk).
                um = getattr(chunk, "usage_metadata", None)
                if um is not None:
                    last_usage = um
                candidates = getattr(chunk, "candidates", []) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if content is None:
                        continue
                    parts = getattr(content, "parts", []) or []
                    for part in parts:
                        t = getattr(part, "text", None)
                        if t:
                            accumulated_text_parts.append(t)
                            if on_event:
                                on_event({"type": "text_chunk", "text": t})
                        # Preserve original Part for function_call (carries
                        # thought_signature).
                        fc = getattr(part, "function_call", None)
                        if fc:
                            accumulated_parts.append(part)
        except Exception as e:
            raise GeminiError(f"streaming iteration failed: {e}") from e

        # Build a synthesized response object that quacks like the non-stream
        # response: has .candidates[0].content.parts[] and .usage_metadata.
        return _StreamedResponse(
            parts=accumulated_parts,
            text="".join(accumulated_text_parts),
            usage_metadata=last_usage,
            raw=last_response,
        )

    # --- Tool dispatch (built-in + MCP + subagent) -----------------------

    def _dispatch_tool_call(
        self,
        *,
        tools: List[Tool],
        fc: Dict[str, Any],
        mcp_client: Optional[Any],
        skills_loader: Optional[Any],
    ) -> Dict[str, Any]:
        """Route a function call to the right executor.

        Order:
            1. Built-in tool with the same name -> execute_tool_call()
            2. MCP tool (name starts with `mcp_`) -> mcp_client.call_tool()
            3. The subagent tool (handled by its own executor)
            4. Unknown -> error response
        """
        name = fc.get("name", "")
        args = fc.get("args") or {}

        # 1. Built-in / subagent tool
        from .tools import find_tool, execute_tool_call
        tool = find_tool(tools, name)
        if tool is not None:
            return execute_tool_call(tools, fc)

        # 2. MCP tool
        if name.startswith("mcp_") and mcp_client is not None:
            result_text = mcp_client.call_tool(name, args)
            response: Dict[str, Any] = {
                "name": name,
                "response": {"result": result_text},
            }
            if fc.get("id"):
                response["id"] = fc["id"]
            return {"function_response": response}

        # 3. Unknown
        response = {
            "name": name,
            "response": {"result": f"Error: unknown tool '{name}'"},
        }
        if fc.get("id"):
            response["id"] = fc["id"]
        return {"function_response": response}

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


def _build_subagent_tool(skills_loader: Optional[Any]) -> Optional[Tool]:
    """Construct the `delegate_subagent` tool, gated on available skills.

    Returns None if no skills are loaded (so the model never sees a tool
    that has nothing to delegate to).
    """
    if skills_loader is None:
        return None
    try:
        enabled = skills_loader.enabled()
    except Exception:
        return None
    if not enabled:
        return None

    # Build a comma-separated list of skill names for the description.
    names = ", ".join(s.name for s in enabled)

    def _exec(skill: str, task: str, **_):
        # Imported here to avoid circular imports at module load time.
        from .skills.subagent import delegate_to_subagent
        # Reuse the singleton client if available, else build a throwaway one.
        # We use a global "last client" reference (set by GeminiClient.__init__).
        client_ref = _LAST_CLIENT_REF[0]
        if client_ref is None:
            return "Error: no parent GeminiClient available for subagent delegation."
        result = delegate_to_subagent(
            parent_client=client_ref,
            skill_name=skill,
            task=task,
            skills_loader=skills_loader,
        )
        if not result.success:
            return f"Subagent error ({skill}): {result.error}"
        text = result.text
        if result.tool_results:
            text += "\n\n(subagent tool calls: " + ", ".join(
                tr["name"] for tr in result.tool_results
            ) + ")"
        return text

    return Tool(
        name="delegate_subagent",
        description=(
            "Delegate a focused task to a specialized subagent driven by a "
            f"skill. Available skills: {names}. The subagent runs in its "
            "own agent turn and returns its final answer."
        ),
        parameters={
            "skill": {
                "type": "string",
                "description": "The skill name to activate for the subagent.",
                "required": True,
            },
            "task": {
                "type": "string",
                "description": "The task to delegate to the subagent.",
                "required": True,
            },
        },
        executor=_exec,
    )


# Module-level singleton reference to the last-constructed GeminiClient.
# Used by the subagent tool to spawn a child turn without re-auth.
_LAST_CLIENT_REF: List[Any] = [None]

# Response-shape helpers live in gemini_py/_response.py and are imported
# here so the agent loop can build synthesized responses when streaming.
from ._response import (  # noqa: E402,F401
    _StreamedResponse,
    _Candidate,
    _Content,
    _TextPart,
    _FunctionCall,
    _FunctionCallPart,
    _UsageMetadata,
    text_response,
    text_chunk,
)


def run_headless(prompt: str, **kwargs) -> GenerateResult:
    """Module-level convenience for `GeminiClient().run_headless`.

    Useful for one-off scripts:

        from gemini_py import run_headless
        print(run_headless("Summarize this code.").text)
    """
    client = GeminiClient()
    return client.run_headless(prompt, **kwargs)
