"""`--help` must not crash.

argparse runs every help string through `help % params` when it formats the
usage screen, so a literal `%` in a help string raises
`TypeError: not enough arguments for format string` -- and only ever at
`--help` time, never during a normal run. Three help strings quote measured
percentages (the --push-top breadth numbers, the --fetch-fx currency share),
which is exactly the kind of text that keeps re-introducing this. Literal
percent signs must be written `%%`.

Guards the crash and the escaping in one go: if `%%` were over-corrected to a
literal `%%` in the output, or a real `%` slipped back in, one of these fails.
"""
import pytest

from analyzer.main import build_arg_parser


def test_help_renders_without_format_error():
    parser = build_arg_parser()
    text = parser.format_help()
    assert "usage:" in text


def test_escaped_percentages_render_as_single_percent():
    text = build_arg_parser().format_help()
    # Collapse argparse's line wrapping before matching.
    flat = " ".join(text.split())
    assert "9.98% at N=10" in flat
    assert "4.48% at N=25" in flat
    assert "44% of the universe" in flat
    assert "%%" not in flat


def test_every_help_string_survives_percent_expansion():
    """Catches a bad `%` in any *future* help string, not just today's three."""
    parser = build_arg_parser()
    for action in parser._actions:
        if not action.help:
            continue
        try:
            action.help % {"default": action.default, "prog": parser.prog}
        except (TypeError, ValueError, KeyError) as exc:
            pytest.fail(f"{action.option_strings or action.dest}: {exc}")
