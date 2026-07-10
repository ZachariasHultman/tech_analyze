"""Regression tests for _resolve_universe (analyzer/main.py): a bare
invocation (no --preset, no --watchlists) previously fell back to just the
"Test" watchlist -- a much smaller scope than the standard universe the
weekly cron job actually uses. Now it defaults to the same universe as
cron_pi.sh, but only when BOTH flags are omitted -- giving either one
explicitly must still use exactly what was passed, no default mixed in.
"""
from analyzer.main import _resolve_universe, _DEFAULT_PRESETS, _DEFAULT_WATCHLISTS


def test_both_omitted_uses_default_universe():
    presets, watchlists = _resolve_universe(None, None)
    assert presets == _DEFAULT_PRESETS
    assert watchlists == _DEFAULT_WATCHLISTS


def test_explicit_preset_only_does_not_pull_in_default_watchlists():
    presets, watchlists = _resolve_universe(["omxs30"], None)
    assert presets == ["omxs30"]
    assert watchlists == []


def test_explicit_watchlists_only_does_not_pull_in_default_presets():
    presets, watchlists = _resolve_universe(None, ["Foo"])
    assert presets == []
    assert watchlists == ["Foo"]


def test_both_explicit_uses_exactly_what_was_passed():
    presets, watchlists = _resolve_universe(["omxs30"], ["Foo", "Bar"])
    assert presets == ["omxs30"]
    assert watchlists == ["Foo", "Bar"]
