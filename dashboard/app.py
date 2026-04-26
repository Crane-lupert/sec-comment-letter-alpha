"""Streamlit dashboard — SEC Comment Letter Cross-Section Alpha (Day 8).

Run locally:
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py

Tabs:
  - Overview          headline numbers + pre-registration trail
  - Topic heatmap     14 topics x 4 severity bands, color = mean BHAR
  - Severity dist     per-severity-band CAR/BHAR distribution
  - Cum returns       Signal A vs B vs FF5+UMD baseline cumulative
  - Robustness        Day 7 size/sector/liquidity stratified alpha
  - TC sensitivity    post-cost Sharpe at 5/10/20 bps/month
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    raise SystemExit("streamlit + plotly not available; install via `uv pip install streamlit plotly`")

REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED = Path(__file__).resolve().parent / "data_for_app"
DATA = _BUNDLED if _BUNDLED.exists() else REPO_ROOT / "data"

st.set_page_config(page_title="SEC Comment Letter Alpha", layout="wide")

# ----------------------------- Data loaders -----------------------------

@st.cache_data
def load_pairs() -> pd.DataFrame:
    p = DATA / "day4_pairs.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_json(p, lines=True)


@st.cache_data
def load_factor_returns(matched: bool = False) -> pd.DataFrame:
    p = DATA / ("day6_factor_returns_matched.parquet" if matched else "day4_factor_returns.parquet")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_alpha_summary(matched: bool = False) -> dict:
    p = DATA / ("day6_alpha_summary_matched.json" if matched else "day4_alpha_summary.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_tc_summary() -> dict:
    p = DATA / "day6_post_tc_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_robust_summary() -> dict:
    p = DATA / "day7_robustness_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_events() -> pd.DataFrame:
    p = DATA / "day4_events.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ----------------------------- Sidebar nav -----------------------------

st.sidebar.title("SEC Comment Letter Alpha")
st.sidebar.caption("Project X — QR Scout Portfolio")
st.sidebar.markdown("---")
use_matched = st.sidebar.toggle("Use Day 6 matched control", value=True,
                                 help="Day 4 used sector-mean of recipients (over-stated). Day 6 uses K=5 matched non-letter R3K firms.")
view = st.sidebar.radio("View", [
    "Overview", "Topic heatmap", "Severity distribution",
    "Cumulative returns", "Robustness", "TC sensitivity", "About",
])

pairs = load_pairs()
fac_rets = load_factor_returns(matched=use_matched)
alpha = load_alpha_summary(matched=use_matched)
tc = load_tc_summary()
robust = load_robust_summary()
events = load_events()

# ----------------------------- Views -----------------------------

if view == "Overview":
    st.title("Cross-Section Alpha from SEC Comment Letters")
    st.caption("Pre-registered cross-section factor on Russell 3000 UPLOAD-CORRESP pair events. "
               "Source: SEC EDGAR via daemon-mediated cache. LLM ensemble (gemma-3-27b + llama-3.3-70b "
               "+ claude-opus-4.7 oracle) for letter feature extraction.")

    c1, c2, c3, c4 = st.columns(4)
    if pairs.empty:
        st.warning("No pair data found. Run scripts/day4_run_all.py first.")
    else:
        c1.metric("Pairs (R3K)", f"{len(pairs):,}")
        in_target = (
            pd.to_datetime(pairs["upload_date"]).dt.year.between(2015, 2024).sum()
            if "upload_date" in pairs else 0
        )
        c2.metric("In 2015-2024 window", f"{in_target:,}")
        agreed = pairs.get("corresp_response_intent_agreed", pd.Series([])).sum()
        c3.metric("v3-corresp intent agreed", f"{agreed:,}")
        if "topic_match_jaccard" in pairs:
            c4.metric("Mean topic-match Jaccard", f"{pairs['topic_match_jaccard'].mean():.2f}")

    st.markdown("---")
    st.subheader("Pre-registered main cell — BHAR t+1..t+60, FF5+UMD residual")

    if alpha and "signals" in alpha:
        rows = []
        # Matched alpha summary uses keys like 'A_bhar_2m_matched';
        # sector-mean (Day 4) summary uses 'A_bhar_2m'. Look in both.
        candidate_sigs = ["A_bhar_2m_matched", "B_bhar_2m_matched",
                          "A_bhar_2m", "B_bhar_2m"]
        signals_present = alpha.get("signals", {})
        for sig in candidate_sigs:
            if sig not in signals_present:
                continue
            display = sig.replace("_matched", " (matched)")
            for w in ["FULL", "IS_2015_2021", "OOS_2022_2024"]:
                b = signals_present[sig].get(w, {})
                if not b.get("alpha"):
                    continue
                rows.append({
                    "Signal": display, "Window": w,
                    "n_months": b.get("n_months", "—"),
                    "Sharpe (raw)": round(b.get("raw_sharpe_annual", float("nan")), 2),
                    "α annual": f"{b['alpha'].get('alpha_annual', 0)*100:+.2f}%",
                    "t-stat": round(b["alpha"].get("t_alpha", float("nan")), 2),
                    "p-value": round(b["alpha"].get("p_alpha", float("nan")), 3),
                    "DSR": round(b.get("deflated_sharpe_ratio", float("nan")), 2),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("**Pre-registration trail**: commits `c4cf77b` (Day 4 spec), `23cabfe` (CORRESP v3-corresp schema), `0819f75` (held-out validation). All locked BEFORE the corresponding extraction.")
    st.markdown("**Contamination audit**: gemma κ=1.000, llama κ=1.000 on n=50 redacted-vs-original. LLM is reading text, not recalling firm-specific outcomes.")
    st.markdown("**Limitations**: see `docs/limitations.md`. Day 4 IS alpha was largely sector-residualization artifact — Day 6 matched control reveals the OOS-only nature of the signal.")


elif view == "Topic heatmap":
    st.title("Topic × Severity heatmap")
    st.caption("Mean Signal-A BHAR (2m forward, sector-mean control) per topic and severity band. "
               "Cell color: blue = letter recipients underperformed (positive short-side return).")

    if events.empty:
        st.warning("Events not loaded.")
    else:
        df = events.copy()
        df["sev_band"] = pd.cut(df["upload_severity_mean"],
                                  bins=[0, 0.2, 0.5, 0.8, 1.01],
                                  labels=["0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1.0"])
        # explode topics. The column may be a numpy.ndarray (parquet
        # round-trip), a list, None, or a NaN scalar — handle all four.
        rows = []
        for _, r in df.iterrows():
            topics = r.get("upload_topics_consensus")
            if topics is None:
                continue
            try:
                items = list(topics)
            except TypeError:
                continue
            for t in items:
                rows.append({"topic": t, "sev_band": r["sev_band"], "bhar_2m": r.get("bhar_a_2m")})
        if rows:
            sub = pd.DataFrame(rows).dropna(subset=["bhar_2m", "topic", "sev_band"])
            grid = sub.groupby(["topic", "sev_band"], observed=True)["bhar_2m"].agg(["mean", "count"]).reset_index()
            pivot_mean = grid.pivot(index="topic", columns="sev_band", values="mean")
            pivot_n = grid.pivot(index="topic", columns="sev_band", values="count")
            fig = px.imshow(pivot_mean, color_continuous_scale="RdBu_r", aspect="auto",
                            origin="lower", zmin=-0.10, zmax=0.10,
                            labels={"color": "mean BHAR 2m"})
            fig.update_layout(height=520, title="Mean recipient BHAR (2m) by topic × severity")
            st.plotly_chart(fig, width="stretch")
            st.caption("Counts:")
            st.dataframe(pivot_n.fillna(0).astype(int), width="stretch")


elif view == "Severity distribution":
    st.title("Severity distribution & per-band BHAR")
    if events.empty:
        st.warning("Events not loaded.")
    else:
        df = events.copy()
        df["sev_band"] = pd.cut(df["upload_severity_mean"],
                                  bins=[0, 0.2, 0.5, 0.8, 1.01],
                                  labels=["0-0.2", "0.2-0.5", "0.5-0.8", "0.8-1.0"])
        c1, c2 = st.columns(2)
        c1.subheader("Severity band counts")
        c1.bar_chart(df["sev_band"].value_counts().sort_index())
        c2.subheader("Mean BHAR (2m) by band")
        c2.bar_chart(df.groupby("sev_band", observed=True)["bhar_a_2m"].mean())


elif view == "Cumulative returns":
    st.title("Cumulative long-short returns")
    if fac_rets.empty:
        st.warning("Factor returns not loaded.")
    else:
        sigs = sorted(fac_rets["signal_id"].unique())
        chosen = st.multiselect("Signals", sigs, default=[s for s in sigs if "bhar_2m" in s])
        sel = fac_rets[fac_rets["signal_id"].isin(chosen)].copy()
        sel["month"] = pd.to_datetime(sel["month"])
        sel = sel.sort_values(["signal_id", "month"])
        sel["cum_log"] = sel.groupby("signal_id")["raw_return"].cumsum()
        fig = px.line(sel, x="month", y="cum_log", color="signal_id",
                      title="Cumulative log return (long-short)")
        st.plotly_chart(fig, width="stretch")


elif view == "Robustness":
    st.title("Day 7 — Robustness across sector / size / liquidity strata")
    if not robust:
        st.warning("Day 7 robustness not run.")
    else:
        rows = []
        for label, b in robust.get("strata", {}).items():
            if b.get("skipped"):
                rows.append({"stratum": label, "n_events": b.get("n_events", "?"),
                             "n_months": "—", "sharpe": "—", "α_annual": "—",
                             "t": "—", "p": "skip"})
            else:
                rows.append({
                    "stratum": label, "n_events": b["n_events"], "n_months": b["n_months"],
                    "sharpe": round(b["sharpe_annual"], 2),
                    "α_annual": f"{b['alpha_annual']*100:+.2f}%",
                    "t": round(b["t_alpha"], 2), "p": round(b["p_alpha"], 3),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Note: signal is concentrated in size_q1-q2 (mid-cap). Largest quintile (q4) shows zero alpha. Smallest (q0) reverses with high noise.")


elif view == "TC sensitivity":
    st.title("Day 6 — Post-transaction-cost sensitivity")
    if not tc:
        st.warning("TC summary not loaded.")
    else:
        sigs = sorted(tc.get("signals", {}).keys())
        chosen = st.selectbox("Signal", sigs, index=sigs.index("B_bhar_2m") if "B_bhar_2m" in sigs else 0)
        rows = []
        for w in ["FULL", "IS_2015_2021", "OOS_2022_2024"]:
            for tc_label in ["raw", "optimistic_5bp", "reasonable_10bp", "conservative_20bp"]:
                key = f"{w}_{tc_label}"
                b = tc["signals"][chosen].get(key)
                if not b: continue
                a = b.get("alpha_post_tc", {})
                rows.append({
                    "window": w, "tc": tc_label,
                    "Sharpe": round(b.get("sharpe_annual_post_tc", float("nan")), 2),
                    "α/yr": f"{a.get('alpha_annual', 0)*100:+.2f}%",
                    "t": round(a.get("t_alpha", float("nan")), 2),
                    "p": round(a.get("p_alpha", float("nan")), 3),
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        be = tc["signals"][chosen].get("break_even_tc_bps", float("nan"))
        st.metric("Break-even TC (bps/month for α=0)", f"{be:.1f} bps/mo")


elif view == "About":
    st.title("About this project")
    st.markdown((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    st.markdown("---")
    st.subheader("Limitations (running ledger)")
    st.markdown((REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8"))
