"""Day 7 — Per-stratum alpha tests with Benjamini-Hochberg FDR correction.

CLAUDE.md Day 6-7 Rigor Checklist requires multiple-comparison correction. The
Day 4 / Day 6 pipeline only deflates 8 pre-registered cells (DSR via Bailey-
López de Prado). Reviewers will ask: "what if you slice by topic? by severity
band? does any topic alone show real per-topic alpha?" Without FDR that becomes
multiple-comparison hunting.

This script:
1. Re-builds per-event matched-control returns using the SAME matching logic
   as scripts/day6_signal_matched.py (re-imported, NOT modified).
2. For each (stratum, window) cell:
     - filters per-event records to events in the stratum
     - aggregates monthly: severity-weighted short on recipients minus
       eq-weight long on matched controls
     - regresses on FF5 + UMD with Newey-West HAC SE (lag=6)
     - records intercept (alpha), t, p, n_months, annualized Sharpe.
3. Applies BH-FDR at α=0.05 across all eligible cells (n_months ≥ 12).
4. Writes:
     - data/day7_fdr_summary.json (full table + BH summary)
     - docs/day7_fdr.md (top survivors + interpretation; pre-existing rewritten)

Stratum grid (per the Day 7 spec):
- 1 full-sample cell
- 14 by-topic cells (TOPIC_ENUM minus 'other'? — including 'other' for 15)
- 4 by-severity-band cells: [0, 0.2], (0.2, 0.5], (0.5, 0.8], (0.8, 1.0]
- 20 cross cells: top-5 topics × 4 severity bands

Times 3 windows (FULL / IS_2015_2021 / OOS_2022_2024) gives ~117 cells; eligible
ones (≥12 months and ≥15 events) are FDR-corrected together.

Underlying signal: B_bhar_2m_matched (pre-registered headline — corresp_date
event, BHAR, 60-day window, sector+size matched control). We do NOT loop over
all 8 matched signals to keep the multiple-comparison surface auditable; the
strata themselves already produce ~100 tests.

Constraints:
- Re-imports scripts.day6_signal_matched helpers — does NOT modify it.
- No LLM calls.
- Min 15 events per cell to avoid tiny-N noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.day6_signal_matched import (  # noqa: E402  (sys.path edit)
    K_MATCH,
    LOOKAHEAD_MONTHS,
    LOOKBACK_MONTHS,
    SIZE_BAND_FRAC,
    _bhar_window,
    _build_letter_recipients_per_month,
    _build_log_ret_pivot,
    _build_size_proxy,
    _car_window,
    _load_french_market,
    _load_letter_dates,
    _match_event,
)
from sec_comment_letter_alpha.features import TOPIC_ENUM  # noqa: E402

# --- Paths ---
EVENTS = REPO_ROOT / "data" / "day4_events.parquet"
PAIRS = REPO_ROOT / "data" / "day4_pairs.jsonl"
RETURNS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
UNIVERSE = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
FRENCH = REPO_ROOT / "data" / "french_factors_monthly.parquet"
OUT_JSON = REPO_ROOT / "data" / "day7_fdr_summary.json"
OUT_MD = REPO_ROOT / "docs" / "day7_fdr.md"

# --- Spec ---
PRIMARY_SIGNAL = {
    "signal_id": "B_bhar_2m_matched",
    "leg": "b",
    "kind": "bhar",
    "horizon": 2,
    "start_col": "b_start_month",
    "rec_col": "bhar_b_2m",
}
NW_LAG = 6
MIN_EVENTS = 15
MIN_MONTHS = 12
ALPHA_FDR = 0.05

IS_START = "2015-01-01"
IS_END = "2021-12-31"
OOS_START = "2022-01-01"
OOS_END = "2024-12-31"

SEVERITY_BANDS = [
    ("sev_0_0.2",   0.0, 0.2),
    ("sev_0.2_0.5", 0.2, 0.5),
    ("sev_0.5_0.8", 0.5, 0.8),
    ("sev_0.8_1.0", 0.8, 1.0001),
]


def build_per_event_table() -> pd.DataFrame:
    """Build the per-event (recipient_ret, control_ret, sector, severity, topics)
    table for the primary signal. Re-uses scripts/day6_signal_matched matching."""
    print("[day7] loading inputs...")
    ev = pd.read_parquet(EVENTS)
    rets = pd.read_parquet(RETURNS)
    rets["date"] = pd.to_datetime(rets["date"])
    universe = pd.read_parquet(UNIVERSE)
    letters = _load_letter_dates(PAIRS)
    print(f"[day7] events={len(ev)} R3K returns={len(rets)} rows / "
          f"{rets['ticker'].nunique()} tickers, universe={len(universe)} CIKs, "
          f"letters={len(letters)}")

    size_proxy = _build_size_proxy(rets)
    log_pivot = _build_log_ret_pivot(rets)
    mkt_log = _load_french_market()
    ticker_to_cik = dict(zip(universe["ticker"], universe["cik"]))
    sector_to_tickers = (
        universe.groupby("sector")["ticker"].apply(list).to_dict()
    )
    all_months = size_proxy.index
    excl = _build_letter_recipients_per_month(letters, all_months)

    leg = PRIMARY_SIGNAL["leg"]
    kind = PRIMARY_SIGNAL["kind"]
    h = PRIMARY_SIGNAL["horizon"]
    start_col = PRIMARY_SIGNAL["start_col"]
    rec_col = PRIMARY_SIGNAL["rec_col"]

    rows = []
    drops = {"no_recipient_ret": 0, "no_match": 0, "no_size_month": 0,
             "letter_no_size": 0, "letter_size_nan": 0,
             "matched_returns_empty": 0}

    for _, row in ev.iterrows():
        recipient_ret = row.get(rec_col)
        if recipient_ret is None or (isinstance(recipient_ret, float) and np.isnan(recipient_ret)):
            drops["no_recipient_ret"] += 1
            continue
        event_month = pd.Timestamp(row[start_col])
        ticker = row["ticker"]
        sector = row["sector"]
        cik = row["cik"]
        excluded = excl.get(event_month + pd.offsets.MonthEnd(0), set()) | {cik}
        matched, why = _match_event(
            ticker, sector, event_month, sector_to_tickers,
            excluded, ticker_to_cik, size_proxy)
        if not matched:
            drops[why if why else "no_match"] = drops.get(why if why else "no_match", 0) + 1
            continue
        ctrl_rets = []
        for mt in matched:
            r = (_bhar_window if kind == "bhar" else _car_window)(
                log_pivot, mkt_log, mt, event_month, h)
            if r is not None:
                ctrl_rets.append(r)
        if not ctrl_rets:
            drops["matched_returns_empty"] += 1
            continue
        ctrl_ret = float(np.mean(ctrl_rets))
        topics = list(row["upload_topics_consensus"]) if row["upload_topics_consensus"] is not None else []
        rows.append({
            "month": event_month,
            "sector": sector,
            "ticker": ticker,
            "cik": cik,
            "severity": float(row["upload_severity_mean"]) if not pd.isna(row["upload_severity_mean"]) else 0.0,
            "topics": topics,
            "recipient_ret": float(recipient_ret),
            "control_ret": ctrl_ret,
            "n_controls": len(ctrl_rets),
        })
    print(f"[day7] per-event records kept: {len(rows)} / {len(ev)} (drops={drops})")
    return pd.DataFrame(rows)


def aggregate_monthly(per_event: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-event records to month-wise long-short raw_return.
    Severity-weighted short on recipients minus eq-weight long on matched controls."""
    if per_event.empty:
        return pd.DataFrame(columns=["month", "raw_return", "n_short"])
    out_rows = []
    for month, grp in per_event.groupby("month"):
        if len(grp) < 2:
            continue
        w = grp["severity"].values
        if w.sum() <= 0:
            w = np.ones(len(grp)) / len(grp)
        else:
            w = w / w.sum()
        short_ret = float((w * grp["recipient_ret"].values).sum())
        long_ret = float(grp["control_ret"].mean())
        net = short_ret - long_ret
        out_rows.append({"month": pd.Timestamp(month), "raw_return": net,
                         "n_short": int(len(grp))})
    if not out_rows:
        return pd.DataFrame(columns=["month", "raw_return", "n_short"])
    return pd.DataFrame(out_rows)


def regress_on_ff5_umd(monthly: pd.DataFrame, french: pd.DataFrame) -> dict:
    """Regress raw_return on FF5+UMD with Newey-West HAC (lag=6).
    Returns dict with alpha_monthly, alpha_annual, t, p, n_months, sharpe."""
    if monthly.empty or len(monthly) < MIN_MONTHS:
        return {"n_months": int(len(monthly)),
                "alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "se_monthly": float("nan"), "t": float("nan"), "p": float("nan"),
                "sharpe_annual": float("nan"), "raw_mean_monthly": float("nan")}
    fdf = french.copy()
    fdf["month"] = pd.to_datetime(fdf["date"]) + pd.offsets.MonthEnd(0)
    fdf = fdf[["month", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]]
    merged = monthly.merge(fdf, on="month", how="inner").sort_values("month").reset_index(drop=True)
    if len(merged) < MIN_MONTHS:
        return {"n_months": int(len(merged)),
                "alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "se_monthly": float("nan"), "t": float("nan"), "p": float("nan"),
                "sharpe_annual": float("nan"), "raw_mean_monthly": float("nan")}
    y = merged["raw_return"].values
    X = sm.add_constant(merged[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]].values)
    try:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
        alpha = float(model.params[0])
        se = float(model.bse[0])
        t = alpha / se if se else float("nan")
        p = float(model.pvalues[0])
    except Exception as e:  # noqa: BLE001
        return {"n_months": int(len(merged)), "error": f"{type(e).__name__}: {e}",
                "alpha_monthly": float("nan"), "alpha_annual": float("nan"),
                "se_monthly": float("nan"), "t": float("nan"), "p": float("nan"),
                "sharpe_annual": float("nan"), "raw_mean_monthly": float("nan")}
    sharpe = float(y.mean() / y.std() * np.sqrt(12)) if y.std() > 0 else float("nan")
    return {
        "n_months": int(len(merged)),
        "alpha_monthly": alpha,
        "alpha_annual": alpha * 12,
        "se_monthly": se,
        "t": t,
        "p": p,
        "sharpe_annual": sharpe,
        "raw_mean_monthly": float(y.mean()),
    }


def filter_by_topic(per_event: pd.DataFrame, topic: str) -> pd.DataFrame:
    if topic == "ANY":
        return per_event
    return per_event[per_event["topics"].apply(lambda ts: topic in ts)].copy()


def filter_by_severity(per_event: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return per_event[(per_event["severity"] >= lo) & (per_event["severity"] < hi)].copy()


def filter_by_window(monthly: pd.DataFrame, window: str) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    if window == "FULL":
        return monthly
    if window == "IS_2015_2021":
        return monthly[(monthly["month"] >= IS_START) & (monthly["month"] <= IS_END)].copy()
    if window == "OOS_2022_2024":
        return monthly[(monthly["month"] >= OOS_START) & (monthly["month"] <= OOS_END)].copy()
    raise ValueError(f"unknown window {window!r}")


def benjamini_hochberg(p_values: list[float], alpha: float = ALPHA_FDR) -> tuple[list[float], list[bool]]:
    """Return (BH-corrected p-values, pass_FDR mask) for a list of p-values.
    NaN inputs map to (NaN, False) and are excluded from the ranking."""
    finite = [(i, p) for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    n = len(finite)
    if n == 0:
        return [float("nan")] * len(p_values), [False] * len(p_values)
    finite_sorted = sorted(finite, key=lambda x: x[1])
    bh = [float("nan")] * len(p_values)
    passes = [False] * len(p_values)
    # BH-adjusted q-value for rank k (1-indexed): q_k = min(p_(j) * n / j) for j >= k.
    # Walk from largest p to smallest, carrying running min.
    running_min = 1.0
    for rank in range(n, 0, -1):
        idx, p = finite_sorted[rank - 1]
        q = p * n / rank
        if q < running_min:
            running_min = q
        bh[idx] = float(running_min)
    # passes: bh[i] <= alpha
    for i, q in enumerate(bh):
        if q is not None and not np.isnan(q) and q <= alpha:
            passes[i] = True
    return bh, passes


# ---------------------------------------------------------------------------
# Stratum grid
# ---------------------------------------------------------------------------


def build_stratum_grid(per_event: pd.DataFrame) -> list[dict]:
    """Return list of cell descriptors (subset_filter applied lazily)."""
    # Top-5 topics by recipient count.
    topic_counts = {}
    for ts in per_event["topics"]:
        for t in ts:
            topic_counts[t] = topic_counts.get(t, 0) + 1
    top5 = [t for t, _ in sorted(topic_counts.items(), key=lambda kv: -kv[1])[:5]]

    cells: list[dict] = []
    cells.append({"name": "ALL", "topic": "ANY", "sev_lo": 0.0, "sev_hi": 1.001})
    for topic in TOPIC_ENUM:  # 15 entries inc. "other"
        cells.append({"name": f"topic={topic}", "topic": topic,
                      "sev_lo": 0.0, "sev_hi": 1.001})
    for label, lo, hi in SEVERITY_BANDS:
        cells.append({"name": label, "topic": "ANY", "sev_lo": lo, "sev_hi": hi})
    for topic in top5:
        for label, lo, hi in SEVERITY_BANDS:
            cells.append({"name": f"topic={topic}|{label}", "topic": topic,
                          "sev_lo": lo, "sev_hi": hi})
    print(f"[day7] stratum grid: {len(cells)} base cells; top-5 topics = {top5}")
    return cells


def evaluate_cell(per_event: pd.DataFrame, french: pd.DataFrame,
                  cell: dict, window: str) -> dict:
    sub = filter_by_topic(per_event, cell["topic"])
    sub = filter_by_severity(sub, cell["sev_lo"], cell["sev_hi"])
    n_events = int(len(sub))
    monthly = aggregate_monthly(sub)
    monthly = filter_by_window(monthly, window)
    n_eligible_events = n_events  # diagnostic
    if n_events < MIN_EVENTS or len(monthly) < MIN_MONTHS:
        return {
            "stratum": cell["name"],
            "window": window,
            "n_events": n_events,
            "n_months": int(len(monthly)),
            "alpha_annual": float("nan"),
            "alpha_monthly": float("nan"),
            "se_monthly": float("nan"),
            "t": float("nan"),
            "p_raw": float("nan"),
            "sharpe_annual": float("nan"),
            "raw_mean_monthly": float("nan"),
            "eligible": False,
            "reason_skipped": ("n_events<15" if n_events < MIN_EVENTS
                               else "n_months<12"),
        }
    res = regress_on_ff5_umd(monthly, french)
    eligible = (res["n_months"] >= MIN_MONTHS) and (n_eligible_events >= MIN_EVENTS) \
               and not np.isnan(res["p"])
    return {
        "stratum": cell["name"],
        "window": window,
        "n_events": n_events,
        "n_months": res["n_months"],
        "alpha_annual": res["alpha_annual"],
        "alpha_monthly": res["alpha_monthly"],
        "se_monthly": res["se_monthly"],
        "t": res["t"],
        "p_raw": res["p"],
        "sharpe_annual": res["sharpe_annual"],
        "raw_mean_monthly": res["raw_mean_monthly"],
        "eligible": bool(eligible),
        "reason_skipped": "" if eligible else "regression_nan",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_markdown(report_path: Path, summary: dict) -> None:
    rows = summary["rows"]
    bh_summary = summary["bh_summary"]
    survivors = sorted(
        [r for r in rows if r.get("passes_FDR_05")],
        key=lambda r: -abs(r["t"]) if r["t"] is not None and not (isinstance(r["t"], float) and np.isnan(r["t"])) else 0
    )
    nominal = [r for r in rows if r.get("eligible") and r["p_raw"] < 0.05]

    def fmt_t(v):
        try:
            return f"{float(v):+.2f}"
        except Exception:
            return "nan"

    def fmt_n(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return "nan"

    lines = []
    lines.append("# Day 7 — Per-stratum FDR analysis\n")
    lines.append("## Why\n")
    lines.append(
        "CLAUDE.md Day 6-7 Rigor Checklist requires Benjamini-Hochberg FDR "
        "correction. Day 4 / Day 6 only deflate the 8 pre-registered cells "
        "(DSR via Bailey-López de Prado). A reviewer can still slice the cohort "
        "by topic or severity band and report a single significant slice — that "
        "is multiple-comparison hunting unless every slice is FDR-controlled. "
        "This document records the corrected exercise.\n"
    )
    lines.append("## Method\n")
    lines.append(
        "1. Underlying signal: `B_bhar_2m_matched` (the pre-registered headline "
        "— corresp_date event, 60-day BHAR, sector+size matched control). The "
        "matched-control build (Day 6) is re-used unchanged via direct import.\n"
        "2. Stratum grid: 1 ALL + 15 topics + 4 severity bands + 5×4 = 20 "
        "topic×band cross-cells = 40 base strata. Each evaluated on FULL / "
        "IS_2015_2021 / OOS_2022_2024 windows = 120 (stratum, window) cells.\n"
        "3. For each cell: filter per-event records, aggregate severity-weighted "
        "long-short, regress monthly returns on FF5+UMD with Newey-West HAC "
        "(lag=6), record alpha+t+p and annualized Sharpe.\n"
        f"4. Eligibility threshold: ≥{MIN_EVENTS} events AND ≥{MIN_MONTHS} months.\n"
        f"5. Apply Benjamini-Hochberg FDR at α={ALPHA_FDR} across ALL eligible "
        "cells jointly.\n"
    )
    lines.append("## Headline numbers\n")
    lines.append(f"- Total cells tested: **{bh_summary['n_total']}**")
    lines.append(f"- Eligible (≥{MIN_EVENTS} events & ≥{MIN_MONTHS} months): **{bh_summary['n_eligible']}**")
    lines.append(f"- Pass nominal p<0.05: **{bh_summary['n_nominal_05']}**")
    lines.append(f"- Pass BH-FDR at α={ALPHA_FDR}: **{bh_summary['n_fdr_05']}**\n")
    lines.append("## Top 10 surviving cells (by |t|)\n")
    if survivors:
        lines.append("| stratum | window | n_months | n_events | alpha_annual | t | p_raw | p_BH |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in survivors[:10]:
            lines.append(
                f"| {r['stratum']} | {r['window']} | {r['n_months']} | {r['n_events']} | "
                f"{fmt_n(r['alpha_annual'])} | {fmt_t(r['t'])} | "
                f"{fmt_n(r['p_raw'])} | {fmt_n(r['p_BH'])} |"
            )
    else:
        lines.append("_No cells survive BH-FDR at α=0.05._\n")
        lines.append("\nAs a consolation, top 10 by nominal-p (for diagnostic only):")
        nom_sorted = sorted(nominal, key=lambda r: -abs(r["t"]) if not np.isnan(r["t"]) else 0)
        if nom_sorted:
            lines.append("\n| stratum | window | n_months | alpha_annual | t | p_raw | p_BH |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in nom_sorted[:10]:
                lines.append(
                    f"| {r['stratum']} | {r['window']} | {r['n_months']} | "
                    f"{fmt_n(r['alpha_annual'])} | {fmt_t(r['t'])} | "
                    f"{fmt_n(r['p_raw'])} | {fmt_n(r['p_BH'])} |"
                )
    lines.append("\n## Interpretation\n")
    if bh_summary["n_fdr_05"] == 0:
        lines.append(
            "**No per-stratum cell survives BH-FDR at α=0.05.** That is the "
            "honest answer: even though some individual slices show nominal "
            "p<0.05, the count is in the range expected by chance for "
            f"{bh_summary['n_eligible']} simultaneous tests. The Day 4/6 "
            "headline alphas are pre-registered (only 8 cells, not 100+); "
            "selecting any one topic ex-post would be data-mining.\n"
        )
        lines.append(
            "This is a *negative result on the topic-stratification dimension*, "
            "not on the main signal. The pre-registered B_bhar_2m_matched alpha "
            "remains the headline; topic-level decomposition adds explanatory "
            "color but no additional tradeable strata.\n"
        )
    else:
        lines.append(
            f"**{bh_summary['n_fdr_05']} cell(s) survive BH-FDR at α=0.05.** "
            "These are robust to multiple-comparison correction across the "
            f"{bh_summary['n_eligible']}-cell stratification grid; alpha here is "
            "not chance-discovered.\n"
        )
        lines.append(
            "Caveat: surviving cells with overlap (e.g. `topic=X` and "
            "`topic=X|sev_band`) are not statistically independent — BH controls "
            "false-discovery rate under arbitrary positive dependence "
            "(Benjamini-Yekutieli would be even more conservative).\n"
        )
    lines.append("## Files\n")
    lines.append("- `scripts/day7_fdr.py` — this analysis\n"
                 "- `data/day7_fdr_summary.json` — full per-cell table + BH summary\n"
                 "- Re-uses (read-only): `scripts/day6_signal_matched.py`, "
                 "`data/day4_events.parquet`, `data/french_factors_monthly.parquet`.\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    per_event = build_per_event_table()
    if per_event.empty:
        print("[day7] per-event table is empty — abort")
        return 1
    french = pd.read_parquet(FRENCH)
    cells = build_stratum_grid(per_event)

    rows = []
    for cell in cells:
        for window in ("FULL", "IS_2015_2021", "OOS_2022_2024"):
            res = evaluate_cell(per_event, french, cell, window)
            rows.append(res)

    eligible_idx = [i for i, r in enumerate(rows) if r["eligible"]]
    p_values = [rows[i]["p_raw"] for i in eligible_idx]
    bh_p, bh_pass = benjamini_hochberg(p_values, alpha=ALPHA_FDR)
    # Default: NaN BH for non-eligible cells
    for r in rows:
        r["p_BH"] = float("nan")
        r["passes_FDR_05"] = False
        r["passes_nominal_05"] = bool(r["eligible"] and not np.isnan(r["p_raw"]) and r["p_raw"] < 0.05)
    for j, i in enumerate(eligible_idx):
        rows[i]["p_BH"] = bh_p[j]
        rows[i]["passes_FDR_05"] = bool(bh_pass[j])

    n_eligible = len(eligible_idx)
    n_nominal = sum(1 for r in rows if r["passes_nominal_05"])
    n_fdr = sum(1 for r in rows if r["passes_FDR_05"])

    summary = {
        "spec": {
            "primary_signal": PRIMARY_SIGNAL["signal_id"],
            "factor_returns_source": "scripts/day6_signal_matched.py (re-used, read-only)",
            "min_events_per_cell": MIN_EVENTS,
            "min_months_per_cell": MIN_MONTHS,
            "newey_west_lag": NW_LAG,
            "fdr_alpha": ALPHA_FDR,
            "fdr_method": "Benjamini-Hochberg",
            "is_window": [IS_START, IS_END],
            "oos_window": [OOS_START, OOS_END],
            "match_K": K_MATCH,
            "match_size_band_frac": SIZE_BAND_FRAC,
            "match_lookback_months": LOOKBACK_MONTHS,
            "match_lookahead_months": LOOKAHEAD_MONTHS,
        },
        "bh_summary": {
            "n_total": len(rows),
            "n_eligible": n_eligible,
            "n_nominal_05": n_nominal,
            "n_fdr_05": n_fdr,
        },
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[day7] wrote {OUT_JSON}")

    write_markdown(OUT_MD, summary)
    print(f"[day7] wrote {OUT_MD}")

    print(f"[day7] cells_total={len(rows)} eligible={n_eligible} "
          f"nominal_p<0.05={n_nominal} BH_FDR_pass={n_fdr}")
    if n_fdr > 0:
        survivors = sorted(
            [r for r in rows if r["passes_FDR_05"]],
            key=lambda r: -abs(r["t"]),
        )
        print("\nTop survivors (BH-FDR pass):")
        for r in survivors[:10]:
            print(f"  {r['stratum']:<35} | {r['window']:<14} | n_m={r['n_months']:>3} "
                  f"n_ev={r['n_events']:>4} | alpha={r['alpha_annual']:+.4f} "
                  f"t={r['t']:+.2f} p={r['p_raw']:.4f} p_BH={r['p_BH']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
