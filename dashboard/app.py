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
def load_factor_returns(matched: bool = False, risk_managed: bool = False) -> pd.DataFrame:
    if risk_managed:
        p = DATA / "day7_risk_managed_factor_returns.parquet"
    elif matched:
        p = DATA / "day6_factor_returns_matched.parquet"
    else:
        p = DATA / "day4_factor_returns.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data
def load_alpha_summary(matched: bool = False, risk_managed: bool = False) -> dict:
    if risk_managed:
        p = DATA / "day7_risk_managed_summary.json"
    elif matched:
        p = DATA / "day6_alpha_summary_matched.json"
    else:
        p = DATA / "day4_alpha_summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_risk_managed_diagnostics() -> dict:
    p = DATA / "day7_risk_managed_diagnostics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_tc_summary(baseline: str = "matched") -> dict:
    """TC sensitivity summary keyed by baseline.

    'sector_mean' -> data/day6_post_tc_summary.json (Day 4 sector-mean control,
                     pre-registered TC table)
    'matched'     -> data/day7_post_tc_matched_summary.json (Day 6 matched,
                     headline-consistent)
    'rm'          -> data/day7_post_tc_rm_summary.json (Day 7 risk-managed N=4)
    """
    name = {
        "sector_mean": "day6_post_tc_summary.json",
        "matched": "day7_post_tc_matched_summary.json",
        "rm": "day7_post_tc_rm_summary.json",
    }.get(baseline, "day7_post_tc_matched_summary.json")
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data
def load_robust_summary(baseline: str = "matched") -> dict:
    """Robustness summary keyed by baseline.

    'sector_mean' -> data/day7_robustness_summary.json (pre-registered, sector-mean)
    'matched'     -> data/day7_robustness_matched_summary.json (Day 6 matched)
    'rm'          -> data/day7_robustness_rm_summary.json (Day 7 risk-managed N=4)
    """
    name = {
        "sector_mean": "day7_robustness_summary.json",
        "matched": "day7_robustness_matched_summary.json",
        "rm": "day7_robustness_rm_summary.json",
    }.get(baseline, "day7_robustness_matched_summary.json")
    p = DATA / name
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
use_risk_managed = st.sidebar.toggle(
    "Risk-managed overlay (Day 7)", value=False,
    help=("Apply A=breadth filter (n_kept>=4), B=per-name 20% cap, C=10% vol-target. "
          "POST-HOC overlay; pre-registered headline is in matched (Day 6) view. "
          "Cutoff N=4 chosen post-sweep over {4,5,6,8} — preserves ~78% of OOS alpha."),
)
# Risk-managed mode requires matched mode (overlay sits on top of Day 6 panel).
rm_engaged = bool(use_matched and use_risk_managed)
if use_risk_managed and not use_matched:
    st.sidebar.warning("Risk-managed overlay requires the matched-control view. "
                       "Falling back to matched (Day 6).")

view_options = [
    "Overview", "Topic heatmap", "Severity distribution",
    "Cumulative returns", "Robustness", "TC sensitivity",
    "Risk-managed comparison", "About",
]
view = st.sidebar.radio("View", view_options)

pairs = load_pairs()
fac_rets = load_factor_returns(matched=use_matched, risk_managed=rm_engaged)
alpha = load_alpha_summary(matched=use_matched, risk_managed=rm_engaged)
# Always keep the matched (Day 6) reference loaded — used for the overlay
# diff in the Cumulative-returns chart and the Risk-managed comparison view.
matched_fac_rets = load_factor_returns(matched=True, risk_managed=False)
matched_alpha = load_alpha_summary(matched=True, risk_managed=False)
rm_diag = load_risk_managed_diagnostics()
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
        # sector-mean (Day 4) summary uses 'A_bhar_2m'; risk-managed uses
        # 'A_bhar_2m_matched_rm'. Look in all three.
        candidate_sigs = ["A_bhar_2m_matched_rm", "B_bhar_2m_matched_rm",
                          "A_bhar_2m_matched", "B_bhar_2m_matched",
                          "A_bhar_2m", "B_bhar_2m"]
        signals_present = alpha.get("signals", {})
        for sig in candidate_sigs:
            if sig not in signals_present:
                continue
            display = (sig.replace("_matched_rm", " (RM)")
                          .replace("_matched", " (matched)"))
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
    if rm_engaged:
        st.info("Risk-managed (RM) view is active. **POST-HOC** overlay (A=breadth>=4, "
                "B=per-name 20% cap, C=10% vol-target). Pre-registered headline α is "
                "the matched (Day 6) row above; toggle off the RM switch to revert.")

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
        # When RM overlay is engaged, also overlay the matched (Day 6) curves
        # in light gray so the visual A/B is immediate.
        if rm_engaged and not matched_fac_rets.empty:
            # Pair each chosen RM signal with its matched counterpart.
            ref_sigs = sorted({s.replace("_rm", "") for s in chosen if s.endswith("_rm")})
            ref = matched_fac_rets[matched_fac_rets["signal_id"].isin(ref_sigs)].copy()
            if not ref.empty:
                ref["month"] = pd.to_datetime(ref["month"])
                ref = ref.sort_values(["signal_id", "month"])
                ref["cum_log"] = ref.groupby("signal_id")["raw_return"].cumsum()
                for sid, grp in ref.groupby("signal_id"):
                    fig.add_trace(go.Scatter(
                        x=grp["month"], y=grp["cum_log"],
                        mode="lines", name=f"{sid} (matched, ref)",
                        line=dict(color="lightgray", width=1.5, dash="dot"),
                        hoverinfo="skip", showlegend=True,
                    ))
        st.plotly_chart(fig, width="stretch")
        if rm_engaged:
            st.caption("Light-gray dotted lines show the matched (Day 6) reference "
                       "for visual A/B. Solid colored lines are the risk-managed (Day 7) "
                       "post-hoc overlay.")


elif view == "Robustness":
    st.title("Day 7 — Robustness across sector / size / liquidity strata")
    rob_baseline_label = st.selectbox(
        "Robustness baseline",
        ["matched (Day 6)", "risk-managed (Day 7, N=4)", "sector-mean (Day 4, original)"],
        index=0,
        help=("Pick the underlying long-short factor whose per-stratum alpha is "
              "shown. matched (Day 6) is the headline-consistent baseline; "
              "sector-mean (Day 4) is the pre-registered original."),
    )
    rob_key = {
        "matched (Day 6)": "matched",
        "risk-managed (Day 7, N=4)": "rm",
        "sector-mean (Day 4, original)": "sector_mean",
    }[rob_baseline_label]
    robust = load_robust_summary(rob_key)
    st.caption(f"Active baseline: {rob_baseline_label}. matched and risk-managed "
               "are new variants for headline-consistent baselines; sector-mean "
               "is the original pre-registered table.")
    if not robust:
        st.warning("Robustness summary not loaded for this baseline.")
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
                    "sharpe": round(b["sharpe_annual"], 2)
                              if pd.notna(b.get("sharpe_annual")) else "—",
                    "α_annual": f"{b['alpha_annual']*100:+.2f}%"
                                if pd.notna(b.get("alpha_annual")) else "—",
                    "t": round(b["t_alpha"], 2)
                         if pd.notna(b.get("t_alpha")) else "—",
                    "p": round(b["p_alpha"], 3)
                         if pd.notna(b.get("p_alpha")) else "—",
                })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if rob_key == "rm":
            st.caption("Note: many strata in the risk-managed view show all-zero "
                        "monthly returns because the breadth filter (n_kept>=4) "
                        "drops every month within that thin stratum.")
        else:
            st.caption("Note: signal is concentrated in size_q1-q2 (mid-cap). "
                        "Largest quintile (q4) shows weak alpha; smallest (q0) "
                        "reverses with high noise.")


elif view == "TC sensitivity":
    st.title("Post-transaction-cost sensitivity")
    tc_baseline_label = st.selectbox(
        "TC baseline",
        ["matched (Day 6)", "risk-managed (Day 7, N=4)", "sector-mean (Day 4, original spec)"],
        index=0,
        help=("Pick the underlying long-short factor whose monthly returns the "
              "TC model deducts from. matched (Day 6) is the headline-consistent "
              "baseline."),
    )
    tc_key = {
        "matched (Day 6)": "matched",
        "risk-managed (Day 7, N=4)": "rm",
        "sector-mean (Day 4, original spec)": "sector_mean",
    }[tc_baseline_label]
    tc = load_tc_summary(tc_key)
    st.caption(f"Active baseline: {tc_baseline_label}. The sector-mean view is "
               "the original pre-registered TC table; matched and risk-managed "
               "are new variants for headline-consistent baselines.")
    if not tc:
        st.warning("TC summary not loaded for this baseline.")
    else:
        sigs = sorted(tc.get("signals", {}).keys())
        # Default to the headline A 2m signal in the active baseline.
        default_sig = next(
            (s for s in ["A_bhar_2m_matched", "A_bhar_2m_matched_rm", "A_bhar_2m"]
             if s in sigs),
            sigs[0] if sigs else "",
        )
        chosen = st.selectbox(
            "Signal", sigs,
            index=sigs.index(default_sig) if default_sig in sigs else 0,
        )
        rows = []
        for w in ["FULL", "IS_2015_2021", "OOS_2022_2024"]:
            for tc_label in ["raw", "optimistic_5bp", "reasonable_10bp", "conservative_20bp"]:
                key = f"{w}_{tc_label}"
                b = tc["signals"][chosen].get(key)
                if not b:
                    continue
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


elif view == "Risk-managed comparison":
    st.title("Day 7 — Risk-managed overlay (post-hoc) vs Day 6 matched")
    st.caption("Overlay A=breadth filter (n_kept>=4), B=per-name 20% cap "
               "(or 1.5/N for small N), C=10% vol-target with 6m lag-1 rolling "
               "sigma. Pre-registered headline alpha is the matched (Day 6) "
               "view; this overlay is POST-HOC. Cutoff N=4 chosen post-sweep "
               "over {4,5,6,8} — sweep results archived in "
               "`data/day7_risk_managed_n{N}_*` files.")

    if not rm_diag:
        st.warning("Day 7 risk-managed diagnostics not found. "
                   "Run scripts/day7_risk_managed_overlay.py first.")
    else:
        spec = rm_diag.get("spec", {})
        cols = st.columns(5)
        cols[0].metric("Breadth min", spec.get("breadth_min", "?"))
        cols[1].metric("Name cap (base)", f"{spec.get('name_cap_base', 0)*100:.0f}%")
        cols[2].metric("Vol target (annual)", f"{spec.get('vol_target_annual', 0)*100:.0f}%")
        cols[3].metric("Vol lookback (months)", spec.get("vol_lookback_months", "?"))
        cols[4].metric("Leverage cap", f"{spec.get('lev_cap', 0):.1f}x")

        st.subheader("MDD before / after")
        mdd_rows = rm_diag.get("mdd_comparison", [])
        if mdd_rows:
            df_mdd = pd.DataFrame([{
                "Signal": r["signal_id"],
                "MDD matched (Day 6)": f"{r['mdd_matched']*100:+.1f}%",
                "MDD risk-managed (Day 7)": f"{r['mdd_risk_managed']*100:+.1f}%",
                "MDD reduction (abs)": f"{r['mdd_improvement_abs']*100:+.1f}pp",
                "Calmar (RM, OOS)": (round(r["calmar_rm_oos"], 2)
                                     if r.get("calmar_rm_oos") is not None else "—"),
            } for r in mdd_rows])
            st.dataframe(df_mdd, hide_index=True, width="stretch")

        st.subheader("Per-signal overlay diagnostics")
        per = rm_diag.get("per_signal", {})
        if per:
            df_per = pd.DataFrame([{
                "Signal": k,
                "Total months": v.get("n_months_total"),
                "Months dropped (n_kept<4)": v.get("n_months_dropped"),
                "Months name-cap hit": v.get("n_months_capped"),
                "Months lev capped (=2x)": v.get("n_months_lev_capped"),
                "Mean n_kept": round(v.get("mean_n_kept", float("nan")), 2),
                "Mean leverage": round(v.get("mean_leverage", float("nan")), 2),
                "Max leverage": round(v.get("max_leverage", float("nan")), 2),
            } for k, v in per.items()])
            st.dataframe(df_per, hide_index=True, width="stretch")

        st.markdown("---")
        st.markdown(
            "**Interpretation note**: with median n_kept = 4-6 across signals, "
            "the cutoff value matters a lot. Initial spec used N=8 (a-priori "
            "guess) and dropped 70% of months — alpha collapsed to +7.3% (t=1.61) "
            "even as MDD fell to -15%. After sweep over {4,5,6,8}, **N=4 was "
            "selected**: drops only 27% of months, preserves OOS α at +20.8%/yr "
            "(t=3.08, p=0.002 — t-stat actually IMPROVES vs raw matched 2.86 "
            "because vol-target shrinks SE faster than alpha), MDD -13.6%, "
            "Calmar 1.53. Per-name cap and vol target are second-order to the "
            "breadth filter. The headline pre-registered alpha remains the Day 6 "
            "matched cell — this overlay is a separate *implementable variant*."
        )


elif view == "About":
    st.title("About this project")
    st.markdown((REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    st.markdown("---")
    st.subheader("Limitations (running ledger)")
    st.markdown((REPO_ROOT / "docs" / "limitations.md").read_text(encoding="utf-8"))
