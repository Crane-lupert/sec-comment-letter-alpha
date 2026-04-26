"""Day 6 matched-control unit tests — pure-Python logic, no I/O on data files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Load day6 module by file path (it's a script, not a package)
REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "day6_signal_matched", REPO_ROOT / "scripts" / "day6_signal_matched.py"
)
day6 = importlib.util.module_from_spec(SPEC)
sys.modules["day6_signal_matched"] = day6
SPEC.loader.exec_module(day6)  # type: ignore[attr-defined]


def _make_size_proxy() -> pd.DataFrame:
    months = pd.date_range("2018-01-31", "2018-06-30", freq="ME")
    data = {
        "AAA": [100.0] * 6,           # letter firm: $100
        "BBB": [110.0] * 6,           # +10% — IN band
        "CCC": [120.0] * 6,           # +20% — IN band (boundary inclusive)
        "DDD": [125.0] * 6,           # +25% — OUT of band
        "EEE": [85.0]  * 6,           # -15% — IN band
        "FFF": [70.0]  * 6,           # -30% — OUT of band
        "GGG": [102.0] * 6,           # +2% — IN band, but in EXCLUDED set
    }
    return pd.DataFrame(data, index=months)


def test_match_event_basic_band():
    sp = _make_size_proxy()
    sector_to_tickers = {"Tech": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]}
    ticker_to_cik = {t: f"cik_{t}" for t in sector_to_tickers["Tech"]}
    excluded = {"cik_AAA"}  # only the event firm itself
    matched, why = day6._match_event(
        letter_ticker="AAA", sector="Tech",
        event_month=pd.Timestamp("2018-04-30"),
        sector_to_tickers=sector_to_tickers,
        cik_set_excluded=excluded,
        ticker_to_cik=ticker_to_cik,
        size_proxy=sp,
        k=5, band=0.20,
    )
    assert why == ""
    # Expected: BBB (10%), CCC (20%), EEE (15%), GGG (2%) — 4 matches in band
    # DDD (25%) and FFF (30%) excluded
    assert set(matched) == {"BBB", "CCC", "EEE", "GGG"}


def test_match_event_excludes_letter_recipients():
    sp = _make_size_proxy()
    sector_to_tickers = {"Tech": ["AAA", "BBB", "CCC", "EEE", "GGG"]}
    ticker_to_cik = {t: f"cik_{t}" for t in sector_to_tickers["Tech"]}
    # Exclude AAA (event firm) AND GGG (it's a recent letter recipient)
    excluded = {"cik_AAA", "cik_GGG"}
    matched, why = day6._match_event(
        letter_ticker="AAA", sector="Tech",
        event_month=pd.Timestamp("2018-04-30"),
        sector_to_tickers=sector_to_tickers,
        cik_set_excluded=excluded,
        ticker_to_cik=ticker_to_cik,
        size_proxy=sp,
        k=5, band=0.20,
    )
    assert why == ""
    assert "GGG" not in matched
    assert set(matched) == {"BBB", "CCC", "EEE"}


def test_match_event_k_cap():
    sp = _make_size_proxy()
    sector_to_tickers = {"Tech": ["AAA", "BBB", "CCC", "EEE", "GGG"]}
    ticker_to_cik = {t: f"cik_{t}" for t in sector_to_tickers["Tech"]}
    excluded = {"cik_AAA"}
    matched, why = day6._match_event(
        letter_ticker="AAA", sector="Tech",
        event_month=pd.Timestamp("2018-04-30"),
        sector_to_tickers=sector_to_tickers,
        cik_set_excluded=excluded,
        ticker_to_cik=ticker_to_cik,
        size_proxy=sp,
        k=2, band=0.20,
    )
    assert why == ""
    assert len(matched) == 2
    # Closest first: GGG (+2%), BBB (+10%)
    assert matched == ["GGG", "BBB"]


def test_match_event_no_match_returns_reason():
    sp = _make_size_proxy()
    # Letter firm has price 100, all other firms outside ±5% band
    sector_to_tickers = {"Tech": ["AAA", "DDD", "FFF"]}
    ticker_to_cik = {t: f"cik_{t}" for t in sector_to_tickers["Tech"]}
    matched, why = day6._match_event(
        letter_ticker="AAA", sector="Tech",
        event_month=pd.Timestamp("2018-04-30"),
        sector_to_tickers=sector_to_tickers,
        cik_set_excluded={"cik_AAA"},
        ticker_to_cik=ticker_to_cik,
        size_proxy=sp,
        k=5, band=0.05,
    )
    assert matched == []
    assert why == "no_match"


def test_letter_recipients_per_month_window():
    letters = pd.DataFrame({
        "cik": ["c1", "c2", "c3"],
        "upload_date": pd.to_datetime(["2018-01-15", "2018-04-10", "2019-06-01"]),
    })
    months = pd.DatetimeIndex([pd.Timestamp("2018-04-30")])
    excl = day6._build_letter_recipients_per_month(letters, months,
                                                    lookback=12, lookahead=3)
    s = excl[months[0]]
    # c1 (2018-01-15) is within [2017-04-30, 2018-07-31] -> included
    # c2 (2018-04-10) is within window -> included
    # c3 (2019-06-01) is too far in the future -> excluded
    assert s == {"c1", "c2"}


def test_bhar_window_basic():
    months = pd.date_range("2018-01-31", "2018-06-30", freq="ME")
    log_pivot = pd.DataFrame({"AAA": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]},
                              index=months)
    mkt_log = pd.Series([0.005, 0.01, 0.015, 0.02, 0.025, 0.03], index=months)
    # 2-month window starting 2018-03-31: AAA returns 0.03 + 0.04 = 0.07,
    # market 0.015 + 0.02 = 0.035, abnormal = 0.035
    bhar = day6._bhar_window(log_pivot, mkt_log, "AAA",
                              pd.Timestamp("2018-03-31"), 2)
    assert bhar is not None
    assert abs(bhar - 0.035) < 1e-9


def test_bhar_window_missing_data():
    months = pd.date_range("2018-01-31", "2018-06-30", freq="ME")
    log_pivot = pd.DataFrame({"AAA": [0.01, 0.02, np.nan, 0.04, 0.05, 0.06]},
                              index=months)
    mkt_log = pd.Series([0.005] * 6, index=months)
    # 2-month window starting 2018-03-31 — AAA value is NaN at first month
    bhar = day6._bhar_window(log_pivot, mkt_log, "AAA",
                              pd.Timestamp("2018-03-31"), 2)
    assert bhar is None
