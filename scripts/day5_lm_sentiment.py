"""Day 5 LM (Loughran-McDonald) sentiment baseline factor.

For each cached 10-K (under sec-data/edgar-raw/10-K/<cik>_*.json):
  1. Tokenize: lowercase, alphabetic-only, split on whitespace.
  2. Compute neg_ratio = #(words in LM negative list) / #(total words).
  3. Optionally also pos, unc, lit ratios.

Each calendar month, sort firms by their most recent neg_ratio into
quintiles; long top quintile (most negative) - short bottom quintile.

Output: data/day5_lm_factor.parquet (monthly long-short return series)

Note: the canonical factor is sometimes long-bottom-short-top depending on
study; here we treat 'high negative tone' as a risk premium / bad-news
signal and short it (high-neg ⇒ underperform later). Day 5 will run both
directions and keep whichever is consistent with prior literature.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
COORD_ROOT = Path("D:/vscode/portfolio-coordination")
LM_NEG = REPO_ROOT / "data" / "lm_negative_words.txt"
LM_POS = REPO_ROOT / "data" / "lm_positive_words.txt"
LM_UNC = REPO_ROOT / "data" / "lm_uncertainty_words.txt"
LM_LIT = REPO_ROOT / "data" / "lm_litigious_words.txt"
R3K = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
RETS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
TENK_DIR = COORD_ROOT / "sec-data" / "edgar-raw" / "10-K"
OUT_FEATURES = REPO_ROOT / "data" / "day5_lm_features.parquet"
OUT_FACTOR = REPO_ROOT / "data" / "day5_lm_factor.parquet"

_WORD = re.compile(r"[A-Z]+")  # uppercase letters; LM dict is uppercase


def _load_word_set(p: Path) -> set[str]:
    return set(p.read_text(encoding="utf-8").upper().split())


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _ratios(text: str, neg: set[str], pos: set[str], unc: set[str], lit: set[str]) -> dict | None:
    text = _strip_html(text or "").upper()
    words = _WORD.findall(text)
    if len(words) < 500:
        return None
    n = len(words)
    return {
        "n_words": n,
        "neg_ratio": sum(1 for w in words if w in neg) / n,
        "pos_ratio": sum(1 for w in words if w in pos) / n,
        "unc_ratio": sum(1 for w in words if w in unc) / n,
        "lit_ratio": sum(1 for w in words if w in lit) / n,
    }


def build_features() -> pd.DataFrame:
    if not TENK_DIR.exists():
        print(f"[lm] {TENK_DIR} missing")
        return pd.DataFrame()
    neg = _load_word_set(LM_NEG)
    pos = _load_word_set(LM_POS)
    unc = _load_word_set(LM_UNC)
    lit = _load_word_set(LM_LIT)
    print(f"[lm] dict sizes: neg={len(neg)} pos={len(pos)} unc={len(unc)} lit={len(lit)}")

    r3k_ciks = set(pd.read_parquet(R3K)["cik"].astype(str))
    rows = []
    for p in TENK_DIR.glob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cik = obj.get("cik", "")
        if cik not in r3k_ciks:
            continue
        for f in obj.get("filings") or []:
            if f.get("form") != "10-K":
                continue
            r = _ratios(f.get("text", ""), neg, pos, unc, lit)
            if r is None:
                continue
            rows.append({
                "cik": cik, "filing_date": f.get("date"),
                "accession": f.get("accession"), **r,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def build_factor(features: pd.DataFrame, rets: pd.DataFrame, signal_col: str = "neg_ratio") -> pd.DataFrame:
    """Monthly long-short factor: long bottom-quintile, short top-quintile of `signal_col`."""
    r3k = pd.read_parquet(R3K)
    cik_to_ticker = dict(zip(r3k["cik"].astype(str), r3k["ticker"].astype(str)))

    features = features.dropna(subset=[signal_col]).copy()
    features["ticker"] = features["cik"].map(cik_to_ticker)
    features = features.dropna(subset=["ticker"]).sort_values(["ticker", "filing_date"])

    months = pd.date_range(rets["date"].min(), rets["date"].max(), freq="ME")
    rows = []
    for m in months:
        latest = (features[features["filing_date"] <= m]
                  .sort_values("filing_date").groupby("ticker", as_index=False).tail(1))
        # Stale: 10-K from > 18 months ago is too stale
        latest = latest[latest["filing_date"] >= m - pd.Timedelta(days=540)]
        if len(latest) < 30:
            continue
        latest["q"] = pd.qcut(latest[signal_col], 5, labels=False, duplicates="drop")
        long_t = latest.loc[latest["q"] == 0, "ticker"].tolist()  # low neg = positive
        short_t = latest.loc[latest["q"] == 4, "ticker"].tolist()  # high neg = negative
        next_month = (m + pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)
        nx = rets[rets["date"] == next_month]
        if nx.empty:
            continue
        long_ret = nx[nx["ticker"].isin(long_t)]["log_ret"].mean()
        short_ret = nx[nx["ticker"].isin(short_t)]["log_ret"].mean()
        if pd.isna(long_ret) or pd.isna(short_ret):
            continue
        rows.append({
            "month": next_month, "ls_return": float(long_ret - short_ret),
            "n_long": len(long_t), "n_short": len(short_t),
        })
    return pd.DataFrame(rows)


def main():
    feat = build_features()
    print(f"[lm] features: {len(feat)} rows ({feat['cik'].nunique() if not feat.empty else 0} unique CIKs)")
    if feat.empty:
        print("[lm] no 10-K text yet -- daemon still fetching. Run after 10-K cache populates.")
        return 1
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(OUT_FEATURES, index=False)
    print(f"[lm] wrote {OUT_FEATURES}")

    rets = pd.read_parquet(RETS)
    rets["date"] = pd.to_datetime(rets["date"])
    factor = build_factor(feat, rets, signal_col="neg_ratio")
    print(f"[lm] factor months: {len(factor)}")
    if not factor.empty:
        factor.to_parquet(OUT_FACTOR, index=False)
        sharpe = factor["ls_return"].mean() / factor["ls_return"].std() * np.sqrt(12)
        print(f"  raw sharpe (low-neg minus high-neg): {sharpe:.2f}, "
              f"mean monthly: {factor['ls_return'].mean()*100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
