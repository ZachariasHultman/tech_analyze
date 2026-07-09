"""Regression tests for item 5: to_yahoo_symbol (analyzer/main.py).

Covers SE/DK/NO/DE mapping and the crash-prone DE branch / unknown
countryCode cases, which previously produced a raw AttributeError or a
silently wrong symbol.
"""
from analyzer.main import to_yahoo_symbol


def _info(symbol, country):
    return {"listing": {"tickerSymbol": symbol, "countryCode": country}}


def test_to_yahoo_symbol_se():
    assert to_yahoo_symbol(_info("ATCO B", "SE")) == "ATCO-B.ST"


def test_to_yahoo_symbol_dk():
    assert to_yahoo_symbol(_info("NOVO B", "DK")) == "NOVO-B.CO"


def test_to_yahoo_symbol_no():
    assert to_yahoo_symbol(_info("EQNR", "NO")) == "EQNR.OL"


def test_to_yahoo_symbol_de_normal():
    assert to_yahoo_symbol(_info("SAP", "DE")) == "SAP.DE"


def test_to_yahoo_symbol_de_no_regex_match_returns_none():
    # symbol with no leading letters (e.g. all digits/punctuation) — the
    # crash case: re.match returns None, old code called .group() on it.
    assert to_yahoo_symbol(_info("123", "DE")) is None


def test_to_yahoo_symbol_unknown_country_returns_none():
    assert to_yahoo_symbol(_info("XYZ", "US")) is None


def test_to_yahoo_symbol_missing_fields_returns_none():
    assert to_yahoo_symbol({"listing": {}}) is None
    assert to_yahoo_symbol({}) is None
