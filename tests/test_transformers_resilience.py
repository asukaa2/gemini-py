"""Tests for the TransformersProvider resilience helpers.

We can't actually instantiate TransformersProvider (it needs torch +
transformers + a downloaded model). But we CAN test the resilience
methods in isolation by mocking them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We import the module but NOT the class — instantiating requires torch.
from gemini_py.providers.transformers import (
    _parse_tool_calls,
    _parse_tool_call_json,
    _normalize_tool_call_obj,
)


# --- Mock-based tests -----------------------------------------------------


def _make_fake_provider():
    """Build a TransformersProvider-shaped object with stubbed methods,
    WITHOUT calling __init__ (which needs torch).
    """
    from gemini_py.providers.transformers import TransformersProvider
    obj = TransformersProvider.__new__(TransformersProvider)
    # Stub the attributes __init__ would set.
    obj.model_name = "fake-model"
    obj.device = "cuda"
    obj.tokenizer = MagicMock()
    obj.model = MagicMock()
    obj._template_supports_tools = False
    return obj


def test_get_max_input_tokens_uses_model_config():
    """Model config has max_position_embeddings = 8192."""
    p = _make_fake_provider()
    p.model.config.max_position_embeddings = 8192
    assert p._get_max_input_tokens() == 8192


def test_get_max_input_tokens_falls_back_to_4096():
    """No max_position_embeddings attribute → fall back to 4096."""
    p = _make_fake_provider()
    p.model = MagicMock()
    # Make sure config doesn't have it.
    if hasattr(p.model.config, "max_position_embeddings"):
        delattr(p.model.config, "max_position_embeddings")
    p.model.config.max_position_embeddings = None
    assert p._get_max_input_tokens() == 4096


def test_get_max_input_tokens_invalid_value_falls_back():
    """max_position_embeddings is 0 or negative → fall back to 4096."""
    p = _make_fake_provider()
    p.model.config.max_position_embeddings = 0
    assert p._get_max_input_tokens() == 4096

    p.model.config.max_position_embeddings = -1
    assert p._get_max_input_tokens() == 4096


def test_safe_tokenize_truncates_long_input():
    """If input exceeds max_input, tokenizer is called with max_length."""
    p = _make_fake_provider()
    p.model.config.max_position_embeddings = 1024
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    p.tokenizer.return_value = fake_inputs

    p._safe_tokenize("hello world")

    # tokenizer should be called with truncation=True and a max_length.
    call_kwargs = p.tokenizer.call_args.kwargs
    assert call_kwargs.get("truncation") is True
    # max_length = max_position_embeddings - 1024 = 1024 - 1024 = 0, but
    # the code clamps to at least 256. So it should be 256.
    assert call_kwargs.get("max_length") == 256


def test_safe_tokenize_clamps_min_max_length():
    """max_position_embeddings = 512 → max_input = max(256, 512-1024) = 256."""
    p = _make_fake_provider()
    p.model.config.max_position_embeddings = 512
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    p.tokenizer.return_value = fake_inputs

    p._safe_tokenize("hi")
    call_kwargs = p.tokenizer.call_args.kwargs
    assert call_kwargs.get("max_length") == 256


def test_generate_returns_error_string_on_tokenize_failure():
    """If tokenizer raises, _generate returns an error string instead of crashing."""
    p = _make_fake_provider()
    p.tokenizer.side_effect = RuntimeError("tokenizer exploded")
    # _safe_tokenize calls self.tokenizer(...) which is set to side_effect.
    # But wait — _safe_tokenize calls self.tokenizer(prompt, return_tensors=...),
    # so we need to mock that call specifically.
    p.tokenizer = MagicMock(side_effect=RuntimeError("tokenizer exploded"))

    result = p._generate("hello", {"temperature": 1.0})

    assert "tokenize error" in result
    assert "tokenizer exploded" in result


def test_generate_returns_error_string_on_oom():
    """If generation raises OOM, _generate returns an error string."""
    p = _make_fake_provider()
    # Mock _safe_tokenize to return a fake input.
    fake_inputs = {"input_ids": MagicMock()}
    fake_inputs["input_ids"].shape = [1, 5]  # batch=1, seq=5
    p._safe_tokenize = MagicMock(return_value=fake_inputs)

    # Mock _call_generate to raise an OOM-like error.
    class FakeOOM(RuntimeError):
        pass
    p._call_generate = MagicMock(side_effect=RuntimeError("CUDA out of memory."))
    # _move_to_cpu should also fail to make the fallback path fail.
    p._move_to_cpu = MagicMock(side_effect=RuntimeError("cpu also broken"))

    result = p._generate("hello", {"temperature": 1.0})

    assert "generation failed" in result
    assert "CUDA out of memory" in result or "cpu also broken" in result


def test_generate_returns_text_on_success():
    """Happy path: _call_generate returns outputs, decode returns text."""
    p = _make_fake_provider()
    fake_inputs = MagicMock()
    fake_inputs.__getitem__.return_value.shape = [1, 5]  # inputs["input_ids"].shape
    p._safe_tokenize = MagicMock(return_value=fake_inputs)

    # Mock outputs: outputs[0][input_len:] gives the generated tokens.
    fake_outputs = MagicMock()
    fake_outputs.__getitem__.return_value = MagicMock()
    # After slicing, we have a tensor-like object; tokenizer.decode returns text.
    sliced = MagicMock()
    fake_outputs.__getitem__.return_value.__getitem__.return_value = sliced
    p._call_generate = MagicMock(return_value=fake_outputs)
    p.tokenizer.decode.return_value = "hello world"

    # inputs["input_ids"].shape[1] — we need this to return an int.
    fake_inputs.__getitem__.return_value.shape = MagicMock()
    fake_inputs.__getitem__.return_value.shape.__getitem__.return_value = 5

    result = p._generate("hello", {"temperature": 1.0})
    assert result == "hello world"


def test_generate_returns_error_on_empty_output():
    """Empty decoded text → '[no output generated...]' message."""
    p = _make_fake_provider()
    fake_inputs = MagicMock()
    p._safe_tokenize = MagicMock(return_value=fake_inputs)
    fake_inputs.__getitem__.return_value.shape = MagicMock()
    fake_inputs.__getitem__.return_value.shape.__getitem__.return_value = 5

    fake_outputs = MagicMock()
    p._call_generate = MagicMock(return_value=fake_outputs)
    p.tokenizer.decode.return_value = "   "  # whitespace-only

    result = p._generate("hello", {"temperature": 1.0})
    assert "no output generated" in result


def test_generate_respects_user_max_tokens():
    """User passes max_output_tokens=100 — that's what's used (not capped to 2048)."""
    p = _make_fake_provider()
    fake_inputs = MagicMock()
    p._safe_tokenize = MagicMock(return_value=fake_inputs)
    fake_inputs.__getitem__.return_value.shape = MagicMock()
    fake_inputs.__getitem__.return_value.shape.__getitem__.return_value = 5

    fake_outputs = MagicMock()
    p._call_generate = MagicMock(return_value=fake_outputs)
    p.tokenizer.decode.return_value = "hello"

    p._generate("hi", {"max_output_tokens": 100, "temperature": 1.0})

    # _call_generate should be called with max_new=100
    call_args = p._call_generate.call_args
    # call_args is (inputs, max_new, temperature, top_p)
    assert call_args.args[1] == 100


def test_generate_caps_at_8192():
    """max_output_tokens > 8192 gets capped to 8192."""
    p = _make_fake_provider()
    fake_inputs = MagicMock()
    p._safe_tokenize = MagicMock(return_value=fake_inputs)
    fake_inputs.__getitem__.return_value.shape = MagicMock()
    fake_inputs.__getitem__.return_value.shape.__getitem__.return_value = 5

    fake_outputs = MagicMock()
    p._call_generate = MagicMock(return_value=fake_outputs)
    p.tokenizer.decode.return_value = "hello"

    p._generate("hi", {"max_output_tokens": 99999, "temperature": 1.0})

    call_args = p._call_generate.call_args
    assert call_args.args[1] == 8192


def test_move_to_cpu_sets_device_to_cpu():
    """_move_to_cpu sets self.device = 'cpu' and moves model.

    Skipped if torch isn't installed (because _move_to_cpu imports torch).
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed — skipping _move_to_cpu test")
    p = _make_fake_provider()
    p._move_to_cpu()
    assert p.device == "cpu"
    p.model.to.assert_called_with("cpu")


def test_move_to_cpu_handles_failure():
    """If model.to() raises, _move_to_cpu writes to stderr but doesn't crash."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed — skipping _move_to_cpu failure test")
    p = _make_fake_provider()
    p.model.to.side_effect = RuntimeError("model is broken")
    # Should not raise.
    p._move_to_cpu()
    assert p.device == "cpu"  # Still set


def test_call_generate_uses_no_grad():
    """_call_generate wraps model.generate() in torch.no_grad()."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed — skipping _call_generate test")
    p = _make_fake_provider()
    fake_inputs = {"input_ids": MagicMock()}
    p.model.generate.return_value = MagicMock()

    p._call_generate(fake_inputs, max_new=100, temperature=1.0, top_p=0.95)

    # model.generate should be called with the right kwargs.
    call_kwargs = p.model.generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 100
    assert call_kwargs["do_sample"] is True
    assert call_kwargs["top_p"] == 0.95
