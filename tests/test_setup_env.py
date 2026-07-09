"""Regression test for item 4: setup_env's AVANZA_* / legacy env var fallback.

Avanza's constructor is monkeypatched to a fake capturing its kwargs, so
this test never touches the real API.
"""
import pytest

import analyzer.main as main


class _FakeAvanza:
    def __init__(self, creds):
        self.creds = creds


def _clear_env(monkeypatch):
    for key in ("AVANZA_USERNAME", "AVANZA_PASSWORD", "AVANZA_TOTP_SECRET",
                "USERNAME", "PASSWORD", "MY_TOTP_SECRET"):
        monkeypatch.delenv(key, raising=False)


def test_setup_env_prefers_avanza_prefixed_vars(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AVANZA_USERNAME", "new_user")
    monkeypatch.setenv("AVANZA_PASSWORD", "new_pass")
    monkeypatch.setenv("AVANZA_TOTP_SECRET", "new_totp")
    monkeypatch.setattr(main, "Avanza", _FakeAvanza)

    result = main.setup_env()
    assert result.creds == {"username": "new_user", "password": "new_pass", "totpSecret": "new_totp"}


def test_setup_env_falls_back_to_legacy_vars(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("USERNAME", "old_user")
    monkeypatch.setenv("PASSWORD", "old_pass")
    monkeypatch.setenv("MY_TOTP_SECRET", "old_totp")
    monkeypatch.setattr(main, "Avanza", _FakeAvanza)

    result = main.setup_env()
    assert result.creds == {"username": "old_user", "password": "old_pass", "totpSecret": "old_totp"}


def test_setup_env_missing_username_raises(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(main, "Avanza", _FakeAvanza)
    with pytest.raises(Exception, match="AVANZA_USERNAME"):
        main.setup_env()
