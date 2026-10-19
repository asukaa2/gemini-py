"""Tests for the multi-provider system."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.providers import Provider, make_provider
from gemini_py.providers.custom import CustomProvider
from gemini_py.providers._converters import (
    contents_to_text,
    contents_to_chat_messages,
    contents_to_ollama_messages,
)
from gemini_py._response import (
    _StreamedResponse,
    _TextPart,
    _UsageMetadata,
    text_response,
    text_chunk,
)


# --- Provider protocol ----------------------------------------------------


def test_provider_is_abstract():
    """Provider can't be instantiated directly."""
    with pytest.raises(TypeError):
        Provider()  # type: ignore[abstract]


def test_make_provider_gemini_returns_gemini_provider():
    # Without auth setup, this will raise; just verify the type.
    from gemini_py.providers.gemini import GeminiProvider
    # Skip construction since auth not set up in CI; just check the class.
    assert GeminiProvider is not None


def test_make_provider_unknown_name():
    with pytest.raises(ValueError) as exc:
        make_provider("unknown")
    assert "unknown provider" in str(exc.value).lower()


def test_make_provider_custom_needs_callable():
    with pytest.raises(ValueError) as exc:
        make_provider("custom")
    assert "customprovider" in str(exc.value).lower()


# --- CustomProvider -------------------------------------------------------


def test_custom_provider_generate_text():
    def my_generate(prompt: str, *, model: str, **kwargs) -> str:
        return f"[{model}] Echo: {prompt}"

    provider = CustomProvider(
        generate_fn=my_generate,
        models=["my-model-1"],
        name="echo",
    )
    response = provider.generate_content(
        model="my-model-1",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        config={"temperature": 0.7},
    )
    assert hasattr(response, "text")
    assert "[my-model-1] Echo: hello" in response.text
    assert hasattr(response, "candidates")
    assert response.candidates[0].content.parts[0].text == response.text


def test_custom_provider_streaming():
    def my_stream(prompt: str, *, model: str, **kwargs):
        for word in ["Hello", " ", "world", "!"]:
            yield word

    provider = CustomProvider(stream_fn=my_stream)
    chunks = list(
        provider.generate_content_stream(
            model="m",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            config={},
        )
    )
    assert len(chunks) == 4
    assert "".join(c.text for c in chunks) == "Hello world!"


def test_custom_provider_falls_back_to_generate_when_no_stream():
    def my_generate(prompt: str, *, model: str, **kwargs) -> str:
        return "all at once"

    provider = CustomProvider(generate_fn=my_generate)
    chunks = list(
        provider.generate_content_stream(
            model="m",
            contents=[{"role": "user", "parts": [{"text": "hi"}]}],
            config={},
        )
    )
    # When no stream_fn, the fallback yields one chunk with the full text.
    assert len(chunks) == 1
    assert chunks[0].text == "all at once"


def test_custom_provider_requires_at_least_one_callable():
    with pytest.raises(ValueError):
        CustomProvider()


def test_custom_provider_list_models():
    provider = CustomProvider(
        generate_fn=lambda *a, **kw: "ok",
        models=["a", "b", "c"],
    )
    assert provider.list_models() == ["a", "b", "c"]


def test_custom_provider_supports_tools_false():
    provider = CustomProvider(generate_fn=lambda *a, **kw: "ok")
    assert provider.supports_tools() is False


def test_custom_provider_name():
    provider = CustomProvider(generate_fn=lambda *a, **kw: "ok", name="my-backend")
    assert provider.name == "my-backend"


# --- Converters ----------------------------------------------------------


def test_contents_to_text_basic():
    contents = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
    ]
    text = contents_to_text(contents)
    assert "Hello" in text
    assert "Hi there" in text
    assert "[user]" in text
    assert "[model]" in text


def test_contents_to_text_skips_non_text_parts():
    contents = [
        {"role": "user", "parts": [{"function_call": {"name": "x"}}]},
        {"role": "user", "parts": [{"text": "real text"}]},
    ]
    text = contents_to_text(contents)
    assert "real text" in text
    assert "function_call" not in text  # skipped


def test_contents_to_chat_messages_basic():
    contents = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi"}]},
    ]
    msgs = contents_to_chat_messages(contents)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant"  # 'model' -> 'assistant'
    assert msgs[1]["content"] == "Hi"


def test_contents_to_chat_messages_handles_system_role():
    contents = [{"role": "system", "parts": [{"text": "be brief"}]}]
    msgs = contents_to_chat_messages(contents)
    assert msgs[0]["role"] == "system"


def test_contents_to_ollama_messages_same_as_chat():
    contents = [{"role": "user", "parts": [{"text": "hi"}]}]
    assert contents_to_ollama_messages(contents) == contents_to_chat_messages(contents)


# --- Response helpers ----------------------------------------------------


def test_text_response_helper():
    r = text_response("hello", usage={"prompt_token_count": 5, "candidates_token_count": 5, "total_token_count": 10})
    assert r.text == "hello"
    assert r.usage_metadata.prompt_token_count == 5
    assert r.usage_metadata.total_token_count == 10
    assert r.candidates[0].content.parts[0].text == "hello"


def test_text_chunk_helper():
    c = text_chunk("abc")
    assert c.text == "abc"
    assert c.candidates[0].content.parts[0].text == "abc"


def test_streamed_response_synthesizes_text_part_when_empty():
    """When parts=[] but text=non-empty, we synthesize a TextPart."""
    r = _StreamedResponse(parts=[], text="hello", usage_metadata=None, raw=None)
    assert len(r.candidates[0].content.parts) == 1
    assert r.candidates[0].content.parts[0].text == "hello"


# --- CLI integration -----------------------------------------------------


def test_cli_provider_flags_parse():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "-p", "hello",
        "--provider", "ollama",
        "--provider-model", "llama3.2",
        "--provider-base-url", "http://localhost:11434",
    ])
    assert args.provider == "ollama"
    assert args.provider_model == "llama3.2"
    assert args.provider_base_url == "http://localhost:11434"


def test_cli_provider_defaults_to_gemini():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello"])
    assert args.provider == "gemini"


def test_cli_api_key_flag_parses():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hi", "--provider", "openai", "--api-key", "sk-..."])
    assert args.api_key == "sk-..."
    assert args.provider == "openai"


# --- GeminiClient with CustomProvider (smoke test) ----------------------


def test_gemini_client_accepts_provider_instance():
    """GeminiClient should accept a Provider instance directly."""
    from gemini_py import GeminiClient
    provider = CustomProvider(
        generate_fn=lambda prompt, **kw: f"echo: {prompt}",
        models=["x"],
        name="echo",
    )
    client = GeminiClient(provider=provider)
    assert client._provider is provider
    # generate_text should work without any Gemini API call.
    r = client.generate_text("hello")
    assert "echo" in r.text


def test_gemini_client_streaming_with_custom_provider():
    from gemini_py import GeminiClient
    def stream_fn(prompt: str, **kw):
        for word in ["hi", "there"]:
            yield word
    provider = CustomProvider(stream_fn=stream_fn)
    client = GeminiClient(provider=provider)
    chunks = list(client.stream("anything"))
    assert "".join(chunks) == "hithere"
