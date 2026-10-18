"""Tests for the login() / logout() / whoami() credentials API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py import login, logout, whoami, LoginError
from gemini_py.login import (
    _mask,
    _env_line_matches,
    _write_env_file,
    USER_CREDENTIALS_FILE,
    PROJECT_CREDENTIALS_FILE,
)
from gemini_py.auth import AuthConfig, AuthMethod


# --- _mask ----------------------------------------------------------------


def test_mask_short():
    assert _mask("") == ""
    assert _mask("abc") == "***"
    assert _mask("12345678") == "********"


def test_mask_long():
    m = _mask("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
    assert m.startswith("AIza")
    assert m.endswith("1234")
    assert "*" in m
    assert len(m) == len("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234")


# --- _env_line_matches ----------------------------------------------------


def test_env_line_matches_simple():
    assert _env_line_matches("GEMINI_API_KEY=abc", {"GEMINI_API_KEY"}) is True


def test_env_line_matches_export():
    assert _env_line_matches("export GEMINI_API_KEY=abc", {"GEMINI_API_KEY"}) is True


def test_env_line_matches_quoted():
    assert _env_line_matches('GEMINI_API_KEY="abc"', {"GEMINI_API_KEY"}) is True


def test_env_line_matches_other_key():
    assert _env_line_matches("OTHER=xyz", {"GEMINI_API_KEY"}) is False


def test_env_line_matches_no_equals():
    assert _env_line_matches("# comment", {"GEMINI_API_KEY"}) is False
    assert _env_line_matches("", {"GEMINI_API_KEY"}) is False


# --- _write_env_file ------------------------------------------------------


def test_write_env_file_creates(tmp_path):
    path = tmp_path / "creds.env"
    _write_env_file(path, {"GEMINI_API_KEY": "abc"})
    assert path.is_file()
    content = path.read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY="abc"' in content


def test_write_env_file_preserves_other_lines(tmp_path):
    path = tmp_path / "creds.env"
    path.write_text('# My notes\nOTHER=keep\nGEMINI_API_KEY="old"\n', encoding="utf-8")
    _write_env_file(path, {"GEMINI_API_KEY": "new"})
    content = path.read_text(encoding="utf-8")
    assert "My notes" in content
    assert "OTHER=keep" in content
    assert 'GEMINI_API_KEY="new"' in content
    assert 'GEMINI_API_KEY="old"' not in content


def test_write_env_file_replaces_multiple(tmp_path):
    path = tmp_path / "creds.env"
    path.write_text("A=1\nB=2\nA=3\n", encoding="utf-8")
    _write_env_file(path, {"A": "new"})
    content = path.read_text(encoding="utf-8")
    # Both old A= lines should be gone, only the new one present.
    assert 'A="new"' in content
    assert "A=1" not in content
    assert "A=3" not in content
    # B should be preserved.
    assert "B=2" in content


# --- login() --------------------------------------------------------------


def test_login_gemini_default(tmp_path, monkeypatch):
    # Redirect HOME so we don't touch the real ~/.gemini-py
    monkeypatch.setenv("HOME", str(tmp_path))
    # Clean env
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(v, raising=False)

    result = login("AIzaSy_test_key_12345")
    assert result.method == "gemini"
    assert result.api_key == "AIzaSy_test_key_12345"
    assert result.scope == "user"
    assert result.project is None
    assert result.location is None

    # File should exist at the expected path.
    expected_path = tmp_path / ".gemini-py" / ".env"
    assert result.credentials_path == expected_path
    assert expected_path.is_file()
    content = expected_path.read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY="AIzaSy_test_key_12345"' in content

    # Env var should be set in current process too.
    assert os.environ.get("GEMINI_API_KEY") == "AIzaSy_test_key_12345"


def test_login_describe_masked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    result = login("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234")
    desc = result.describe()
    assert "AIza" in desc  # prefix visible
    assert "1234" in desc  # suffix visible
    assert "*" in desc     # middle masked
    # Full key should NOT appear in the describe() output.
    assert "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234" not in desc


def test_login_google_requires_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(LoginError) as exc:
        login("google_key", method="google")
    assert "project is required" in str(exc.value).lower()


def test_login_google_with_project(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(v, raising=False)
    result = login(
        "ya29.google_key",
        method="google",
        project="my-gcp-proj",
        location="us-east1",
    )
    assert result.method == "google"
    assert result.project == "my-gcp-proj"
    assert result.location == "us-east1"
    path = tmp_path / ".gemini-py" / ".env"
    content = path.read_text(encoding="utf-8")
    assert 'GOOGLE_API_KEY="ya29.google_key"' in content
    assert 'GOOGLE_CLOUD_PROJECT="my-gcp-proj"' in content
    assert 'GOOGLE_CLOUD_LOCATION="us-east1"' in content
    assert os.environ.get("GOOGLE_API_KEY") == "ya29.google_key"
    assert os.environ.get("GOOGLE_CLOUD_PROJECT") == "my-gcp-proj"


def test_login_unknown_method(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(LoginError):
        login("key", method="invalid")


def test_login_empty_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(LoginError):
        login("")
    with pytest.raises(LoginError):
        login("   ")


def test_login_unknown_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(LoginError):
        login("key", scope="invalid")


def test_login_project_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    result = login("proj_key", scope="project")
    assert result.scope == "project"
    expected_path = tmp_path / ".gemini" / ".env"
    assert result.credentials_path == expected_path
    assert expected_path.is_file()
    content = expected_path.read_text(encoding="utf-8")
    assert 'GEMINI_API_KEY="proj_key"' in content


def test_login_then_authconfig_picks_it_up(tmp_path, monkeypatch):
    """End-to-end: login() writes the file, AuthConfig.from_env() reads it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(v, raising=False)

    # Before login: no creds
    cfg = AuthConfig.from_env(load_dotenv_files=True)
    assert cfg.api_key is None

    # Login
    login("AIza_my_secret_key")
    # AuthConfig should pick it up
    cfg = AuthConfig.from_env(load_dotenv_files=True)
    assert cfg.api_key == "AIza_my_secret_key"
    assert cfg.method == AuthMethod.GEMINI_API_KEY


def test_login_persists_across_calls_in_same_process(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    login("AIza_first")
    assert os.environ.get("GEMINI_API_KEY") == "AIza_first"
    # Login again with a new key — should overwrite
    login("AIza_second")
    assert os.environ.get("GEMINI_API_KEY") == "AIza_second"
    content = (tmp_path / ".gemini-py" / ".env").read_text(encoding="utf-8")
    assert "AIza_first" not in content
    assert "AIza_second" in content


# --- logout() -------------------------------------------------------------


def test_logout_removes_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    login("AIza_then_logout")
    path = tmp_path / ".gemini-py" / ".env"
    assert path.is_file()

    deleted = logout()
    assert deleted is True
    assert not path.is_file()
    # Env vars cleared
    assert os.environ.get("GEMINI_API_KEY") is None


def test_logout_when_not_logged_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    deleted = logout()
    assert deleted is False


def test_logout_unknown_scope():
    with pytest.raises(LoginError):
        logout(scope="invalid")


# --- whoami() -------------------------------------------------------------


def test_whoami_returns_none_when_no_creds(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(v, raising=False)
    assert whoami() is None


def test_whoami_after_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(v, raising=False)
    login("AIza_whoami_test_key_xyz")
    info = whoami()
    assert info is not None
    assert info["method"] == "gemini_api_key"
    assert "AIza" in info["api_key"]
    assert "xyz" in info["api_key"]
    # Full key should be masked.
    assert "AIza_whoami_test_key_xyz" not in info["api_key"]


# --- CLI subcommands ------------------------------------------------------


def test_cli_login_with_api_key_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    from gemini_py.cli import main
    rc = main(["login", "--api-key", "AIza_cli_test_key"])
    assert rc == 0
    # File should exist
    assert (tmp_path / ".gemini-py" / ".env").is_file()
    assert os.environ.get("GEMINI_API_KEY") == "AIza_cli_test_key"


def test_cli_whoami_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(v, raising=False)
    from gemini_py.cli import main
    rc = main(["whoami"])
    assert rc == 0


def test_cli_whoami_after_login(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    from gemini_py.cli import main
    main(["login", "--api-key", "AIza_cli_whoami_key"])
    rc = main(["whoami"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gemini" in out.lower()
    assert "AIza" in out  # prefix visible
    # Full key should not appear in output.
    assert "AIza_cli_whoami_key" not in out


def test_cli_logout_after_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    from gemini_py.cli import main
    main(["login", "--api-key", "AIza_to_logout"])
    rc = main(["logout"])
    assert rc == 0
    assert not (tmp_path / ".gemini-py" / ".env").is_file()


# --- Env loading merge ----------------------------------------------------


def test_env_loads_multiple_files_general_to_specific(tmp_path, monkeypatch):
    """Project .env overrides user .env (specific overrides general)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    for v in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)

    # User-level file (where login() writes)
    user_env = tmp_path / ".gemini-py" / ".env"
    user_env.parent.mkdir(parents=True)
    user_env.write_text('GEMINI_API_KEY="user_key"\nOTHER="user_var"\n', encoding="utf-8")

    # Project-level .env (most specific) — only sets GEMINI_API_KEY
    proj_env = tmp_path / ".env"
    proj_env.write_text('GEMINI_API_KEY="proj_key"\n', encoding="utf-8")

    cfg = AuthConfig.from_env(load_dotenv_files=True)
    # Specific should win — project key.
    assert cfg.api_key == "proj_key"
