"""Tests for the TransformersProvider code-block fallback.

When the model doesn't emit explicit tool-call markers (e.g. base
instruct models without function-calling training), we fall back to
extracting code blocks with file paths and synthesizing write_file
tool calls from them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.providers.transformers import (
    _extract_write_file_from_code_blocks,
    _strip_code_blocks,
    _LANG_TO_EXT,
)


# --- Pattern 1: ```lang:path``` (filename after language) -----------------


def test_extract_lang_path_pattern():
    text = (
        "Here's the file:\n"
        "```python:hello.py\n"
        "print(\"hello\")\n"
        "```\n"
        "Let me know."
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["args"]["path"] == "hello.py"
    assert 'print("hello")' in calls[0]["args"]["content"]


def test_extract_lang_path_with_subdirectory():
    text = (
        "```python:src/main.py\n"
        "print(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "src/main.py"


def test_extract_javascript_file():
    text = (
        "```javascript:app.js\n"
        "console.log('hi');\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "app.js"


# --- Pattern 2: # filename: path (filename in first comment) --------------


def test_extract_filename_comment_pattern():
    text = (
        "Here's the file:\n"
        "```python\n"
        "# filename: src/main.py\n"
        "print(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "src/main.py"


def test_extract_file_prefix_pattern():
    """# file: path also works."""
    text = (
        "```python\n"
        "# file: src/main.py\n"
        "print(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "src/main.py"


def test_extract_path_prefix_pattern():
    """# path: also works."""
    text = (
        "```python\n"
        "# path: src/main.py\n"
        "print(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "src/main.py"


# --- Pattern 3: # path/to/file.ext (no "filename:" prefix) ----------------


def test_extract_path_comment_pattern():
    text = (
        "```js\n"
        "# src/app.js\n"
        "console.log(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "src/app.js"


# --- No path → no calls ----------------------------------------------------


def test_extract_no_path_returns_empty():
    """Plain code block with no path → no calls."""
    text = (
        "Just code:\n"
        "```python\n"
        "print(\"hello\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert calls == []


def test_extract_empty_text():
    assert _extract_write_file_from_code_blocks("") == []


# --- Multiple files --------------------------------------------------------


def test_extract_multiple_files():
    text = (
        "Files:\n"
        "```python:app.py\n"
        "print(1)\n"
        "```\n"
        "And:\n"
        "```js:lib.js\n"
        "console.log(2)\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 2
    paths = [c["args"]["path"] for c in calls]
    assert "app.py" in paths
    assert "lib.js" in paths


def test_extract_skips_duplicate_paths():
    """If the same path appears twice, only the first one is kept."""
    text = (
        "```python:hello.py\n"
        "print(1)\n"
        "```\n"
        "```python:hello.py\n"
        "print(2)\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "hello.py"


# --- Path extension synthesis ---------------------------------------------


def test_extract_adds_extension_when_missing():
    """If the path has no extension, add one based on the language."""
    text = (
        "```python:hello\n"
        "print(\"hi\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    assert calls[0]["args"]["path"] == "hello.py"


def test_extract_javascript_extension():
    text = "```javascript:app\nconsole.log(1)\n```"
    calls = _extract_write_file_from_code_blocks(text)
    assert calls[0]["args"]["path"] == "app.js"


def test_extract_unknown_lang_falls_back_to_txt():
    text = "```unknown:myfile\nsome content\n```"
    calls = _extract_write_file_from_code_blocks(text)
    assert calls[0]["args"]["path"] == "myfile.txt"


# --- _strip_code_blocks ---------------------------------------------------


def test_strip_code_blocks_removes_converted_blocks():
    text = (
        "Here's the file:\n"
        "```python:hello.py\n"
        "print(\"hello\")\n"
        "```\n"
        "Let me know."
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1
    stripped = _strip_code_blocks(text, calls)
    # The code block should be gone, but the surrounding text should remain.
    assert "Here's the file:" in stripped
    assert "Let me know." in stripped
    assert "print(\"hello\")" not in stripped


def test_strip_code_blocks_handles_no_calls():
    """If no calls, the text is returned unchanged (modulo whitespace cleanup)."""
    text = "no code blocks here"
    assert _strip_code_blocks(text, []) == "no code blocks here"


def test_strip_code_blocks_preserves_other_code():
    """Code blocks that weren't converted (no path) should remain."""
    text = (
        "```python:hello.py\n"
        "print(\"hello\")\n"
        "```\n"
        "And some other code:\n"
        "```python\n"
        "print(\"untouched\")\n"
        "```"
    )
    calls = _extract_write_file_from_code_blocks(text)
    assert len(calls) == 1  # only the one with the path
    stripped = _strip_code_blocks(text, calls)
    assert 'print("untouched")' in stripped
    assert 'print("hello")' not in stripped


# --- _LANG_TO_EXT ---------------------------------------------------------


def test_lang_to_ext_has_common_languages():
    assert _LANG_TO_EXT["python"] == "py"
    assert _LANG_TO_EXT["py"] == "py"
    assert _LANG_TO_EXT["javascript"] == "js"
    assert _LANG_TO_EXT["js"] == "js"
    assert _LANG_TO_EXT["typescript"] == "ts"
    assert _LANG_TO_EXT["html"] == "html"
    assert _LANG_TO_EXT["css"] == "css"
    assert _LANG_TO_EXT["json"] == "json"
    assert _LANG_TO_EXT["markdown"] == "md"
    assert _LANG_TO_EXT["bash"] == "sh"
    assert _LANG_TO_EXT["rust"] == "rs"
    assert _LANG_TO_EXT["go"] == "go"
    assert _LANG_TO_EXT["java"] == "java"
    assert _LANG_TO_EXT["c"] == "c"
    assert _LANG_TO_EXT["cpp"] == "cpp"
    assert _LANG_TO_EXT["ruby"] == "rb"
    assert _LANG_TO_EXT["php"] == "php"
