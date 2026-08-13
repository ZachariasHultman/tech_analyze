"""The email the Pi actually sends, rendered end to end.

cron_pi.sh runs `main.py --push --sell-from Äger --email` on the Pi, which
never runs the optimizer -- it only consumes optimization_results_panel.json
copied over from the Mac. These tests exercise that consumption path: the
status block must appear in the sent body, and a Pi holding an older copy of
the file (or none) must still send a complete email rather than crash the
weekly run.
"""

import json

import pytest

import analyzer.main as main_mod
from analyzer.correlation import build_validation_summary


@pytest.fixture
def sent_email(monkeypatch, tmp_path):
    """Capture the body of the email _send_email would transmit."""
    captured = {}

    class _FakeSMTP:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, *a):
            pass

        def sendmail(self, sender, to, msg):
            # The body is a base64 MIME part; assert on the decoded text so
            # these read as tests of what the reader sees.
            import email as email_mod
            parsed = email_mod.message_from_string(msg)
            parts = [p for p in parsed.walk() if p.get_content_type() == "text/plain"]
            captured["raw"] = "\n".join(
                p.get_payload(decode=True).decode("utf-8") for p in parts
            )
            captured["mime"] = msg

    import smtplib
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    for var, val in [("SMTP_USER", "a@b.c"), ("SMTP_PASSWORD", "x"),
                     ("EMAIL_TO", "a@b.c")]:
        monkeypatch.setenv(var, val)
    monkeypatch.setattr(main_mod, "project_root", str(tmp_path))
    return captured, tmp_path


def _write_weights(tmp_path, accept=True, fitted_at=None):
    gate = {
        "accept": accept,
        "n_companies": 127,
        "n_folds_beating_equal": 4,
        "n_folds": 4,
        "permutation_p_value": 0.015,
        "permutation": {"n_permutations": 200},
        "folds": [
            {"fiscal_year": y, "optimized_ic": ic, "optimized_spread": sp,
             "equal_ic": ic - 0.02, "equal_spread": sp - 0.01}
            for y, ic, sp in [(2022, -0.050, -0.052), (2023, 0.081, -0.019),
                              (2024, 0.119, 0.100), (2025, 0.136, 0.076)]
        ],
    }
    validation = build_validation_summary(gate)
    if fitted_at:
        validation["fitted_at"] = fitted_at
    (tmp_path / "optimization_results_panel.json").write_text(json.dumps({
        "optimized_weights": {"roe_pe ratio status": 2.0},
        "optimized_thresholds": {},
        "accepted": accept,
        "dsr": 0.97,
        "confidence": 0.925,
        "validation": validation,
    }))


_PUSH = {
    "target_name": "Bör köpa", "top_n": 10,
    "added": [("Tele2 B 5386", 3.0, 0.81, 0.85, 0.69)],
    "already": [("Veidekke 52602", 8.0, 0.99, 0.98, 0.97)],
    "removed": [], "failed": [],
}


def test_status_block_appears_above_the_buy_list(sent_email):
    captured, tmp_path = sent_email
    _write_weights(tmp_path)
    main_mod._send_email(_PUSH, [])
    body = captured["raw"]

    assert "SYSTEM STATUS" in body
    assert "ACCEPTED" in body
    assert "CONFIDENCE: LOW" in body
    assert "beat equal weight in 4 of 4" in body
    assert "CONSIDER BUYING" in body
    # provenance leads, recommendations follow
    assert body.index("SYSTEM STATUS") < body.index("CONSIDER BUYING")


def test_reject_is_visible_in_the_email(sent_email):
    captured, tmp_path = sent_email
    _write_weights(tmp_path, accept=False)
    main_mod._send_email(_PUSH, [])
    assert "REJECTED" in captured["raw"]


def test_stale_weights_are_flagged(sent_email):
    captured, tmp_path = sent_email
    _write_weights(tmp_path, fitted_at="2019-01-01T00:00:00")
    main_mod._send_email(_PUSH, [])
    assert "STALE" in captured["raw"]


def test_pi_without_the_weights_file_still_sends(sent_email):
    """The weekly run must not die because the SCP was never done."""
    captured, _ = sent_email
    main_mod._send_email(_PUSH, [])
    body = captured["raw"]
    assert "unknown" in body
    assert "CONSIDER BUYING" in body
    assert "Tele2" in body


def test_pi_with_a_legacy_weights_file_still_sends(sent_email):
    captured, tmp_path = sent_email
    (tmp_path / "optimization_results_panel.json").write_text(json.dumps({
        "optimized_weights": {"roe_pe ratio status": 2.0},
        "optimized_thresholds": {}, "accepted": True,
    }))
    main_mod._send_email(_PUSH, [])
    assert "unknown" in captured["raw"]
    assert "CONSIDER BUYING" in captured["raw"]


def test_sell_signals_still_render_alongside_the_block(sent_email):
    captured, tmp_path = sent_email
    _write_weights(tmp_path)
    main_mod._send_email(None, [{"name": "Intrum 5583", "pts": -4.2,
                                 "reasons": "points negative"}])
    body = captured["raw"]
    assert "SYSTEM STATUS" in body
    assert "CONSIDER SELLING" in body
    assert "Intrum" in body
