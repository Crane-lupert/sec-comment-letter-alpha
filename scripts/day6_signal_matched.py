"""Day 6 — Sector-matched non-letter long-control factor returns.

This rebuilds the cross-section signal from `data/day4_events.parquet` with a
proper *matched non-recipient long leg* — replacing the Day 4 short-cut where
the long control was the sector-mean BHAR of letter recipients (i.e. sector
residualization, not a real matched portfolio).

Methodology
-----------
1. Per calendar month M, the "letter-receiving" set is every CIK with an
   `upload_date` in [M - 12 months, M + 3 months]. The clean control universe
   is `R3K \\ letter_set`. The 12-month look-back guarantees recent recipients
   are excluded; the 3-month look-ahead avoids contaminating the control with
   firms that are about to receive a letter.

2. Size proxy. The yfinance returns table has no shares-outstanding history
   (no Compustat dependency by spec) and no volume column. We therefore use
   ``price`` (adjusted close) at the END of the prior month as the simplest
   tradability/size proxy. **Limitation**: price-as-size is correlated with
   share-price *level*, not with market cap. We document this in
   `docs/day6_matched_control.md`.

3. Match. For each event (cik, sector, event_month_start), pick K=5 non-letter
   R3K firms in the same sector whose size-proxy is within ±20% of the letter
   firm's. If <5 match, take all available; if 0 match, drop the event.

4. Per-event matched control BHAR = equal-weight mean BHAR of the K matched
   firms over the same forward window (1m, 2m, 3m).

5. Monthly long-short factor:
       short_leg  = severity-weighted Σ recipient BHAR
       long_leg   = mean of per-event matched-control BHAR (within month)
       net        = short_leg − long_leg

6. Signal IDs: ``A_bhar_{1,2,3}m_matched``, ``B_bhar_{1,2,3}m_matched``,
   ``A_car_2m_matched``, ``B_car_2m_matched`` — same set as Day 4, with
   `_matched` suffix.

Output: ``data/day6_factor_returns_matched.parquet``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS = REPO_ROOT / "data" / "day4_events.parquet"
PAIRS = REPO_ROOT / "data" / "day4_pairs.jsonl"
RETURNS = REPO_ROOT / "data" / "r3k_monthly_returns.parquet"
UNIVERSE = REPO_ROOT / "data" / "universe_ciks_r3k.parquet"
OUT = REPO_ROOT / "data" / "day6_factor_returns_matched.parquet"
OUT_DIAG = REPO_ROOT / "data" / "day6_match_diagnostics.json"

# Match parameters
K_MATCH = 5
SIZE_BAND_FRAC = 0.20  # ±20%
LOOKBACK_MONTHS = 12
LOOKAHEAD_MONTHS = 3

# Same horizons / spec template as Day 4
SIGNAL_SPECS = [
    {"signal_id": "A_bhar_2m_matched", "start_col": "a_start_month", "ret_kind": "bhar", "leg": "a", "horizon": 2},
    {"signal_id": "A_bhar_1m_matched", "start_col": "a_start_month", "ret_kind": "bhar", "leg": "a", "horizon": 1},
    {"signal_id": "A_bhar_3m_matched", "start_col": "a_start_month", "ret_kind": "bhar", "leg": "a", "horizon": 3},
    {"signal_id": "B_bhar_2m_matched", "start_col": "b_start_month", "ret_kind": "bhar", "leg": "b", "horizon": 2},
    {"signal_id": "B_bhar_1m_matched", "start_col": "b_start_month", "ret_kind": "bhar", "leg": "b", "horizon": 1},
    {"signal_id": "B_bhar_3m_matched", "start_col": "b_start_month", "ret_kind": "bhar", "leg": "b", "horizon": 3},
    {"signal_id": "A_car_2m_matched",  "start_col": "a_start_month", "ret_kind": "car",  "leg": "a", "horizon": 2},
    {"signal_id": "B_car_2m_matched",  "start_col": "b_start_month", "ret_kind": "car",  "leg": "b", "horizon": 2},
]


def _load_letter_dates(pairs_path: Path) -> pd.DataFrame:
    """All letter events keyed by cik + upload_date (used for clean-control filter)."""
    rows = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append({"cik": obj["cik"], "upload_date": obj["upload_date"]})
    df = pd.DataFrame(rows)
    df["upload_date"] = pd.to_datetime(df["upload_date"])
    df["upload_month"] = df["upload_date"] + pd.offsets.MonthEnd(0)
    return df


def _bhar_window(rets_pivot_log: pd.DataFrame, mkt_log: pd.Series,
                 ticker: str, start_month: pd.Timestamp, n_months: int) -> float | None:
    """Cum log return of firm minus cum log return of market over the n-month
    window starting at `start_month` inclusive."""
    if ticker not in rets_pivot_log.columns:
        return None
    end_month = (start_month + pd.DateOffset(months=n_months - 1)) + pd.offsets.MonthEnd(0)
    fmask = (rets_pivot_log.index >= start_month) & (rets_pivot_log.index <= end_month)
    firm_ser = rets_pivot_log.loc[fmask, ticker].dropna()
    mmask = (mkt_log.index >= start_month) & (mkt_log.index <= end_month)
    mkt_ser = mkt_log.loc[mmask].dropna()
    if len(firm_ser) < n_months or len(mkt_ser) < n_months:
        return None
    return float(firm_ser.sum() - mkt_ser.sum())


def _car_window(rets_pivot_log: pd.DataFrame, mkt_log: pd.Series,
                ticker: str, start_month: pd.Timestamp, n_months: int) -> float | None:
    """Sum of per-month abnormal log return."""
    if ticker not in rets_pivot_log.columns:
        return None
    end_month = (start_month + pd.DateOffset(months=n_months - 1)) + pd.offsets.MonthEnd(0)
    fmask = (rets_pivot_log.index >= start_month) & (rets_pivot_log.index <= end_month)
    firm_ser = rets_pivot_log.loc[fmask, ticker].dropna()
    mmask = (mkt_log.index >= start_month) & (mkt_log.index <= end_month)
    mkt_ser = mkt_log.loc[mmask].dropna()
    common = firm_ser.index.intersection(mkt_ser.index)
    if len(common) < n_months:
        return None
    return float((firm_ser.loc[common] - mkt_ser.loc[common]).sum())


def _build_size_proxy(rets: pd.DataFrame) -> pd.DataFrame:
    """Return a wide table: index = month-end, columns = ticker, values = price.

    The "size proxy" used for matching is the price at the END of the month
    PRIOR to the event month (point-in-time, no look-ahead).
    """
    return rets.pivot(index="date", columns="ticker", values="price").sort_index()


def _build_log_ret_pivot(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.pivot(index="date", columns="ticker", values="log_ret").sort_index()


def _build_market_log(rets: pd.DataFrame) -> pd.Series:
    """Equal-weighted market log return per month (proxy for `mkt_log_ret`).

    Day 4 used `Mkt-RF + RF` from French. Here the per-event control firms must
    be benchmarked against the SAME market series Day 4 used, so we load the
    French RF+MktRF and use that — matching Day 4's `_bhar` exactly.
    """
    raise NotImplementedError("Use _load_french_market instead")


def _load_french_market() -> pd.Series:
    fr = pd.read_parquet(REPO_ROOT / "data" / "french_factors_monthly.parquet")
    fr["date"] = pd.to_datetime(fr["date"])
    fr["mkt_log_ret"] = fr["Mkt-RF"] + fr["RF"]
    out = fr.set_index("date")["mkt_log_ret"]
    # Align to month-end
    out.index = out.index + pd.offsets.MonthEnd(0)
    return out


def _build_letter_recipients_per_month(letters: pd.DataFrame, all_months: pd.DatetimeIndex,
                                       lookback: int = LOOKBACK_MONTHS,
                                       lookahead: int = LOOKAHEAD_MONTHS) -> dict[pd.Timestamp, set[str]]:
    """For each month M, the set of CIKs that received a letter within
    [M - lookback months, M + lookahead months]. These are EXCLUDED from
    the clean-control universe at month M.
    """
    out: dict[pd.Timestamp, set[str]] = {}
    for m in all_months:
        lo = m - pd.DateOffset(months=lookback)
        hi = m + pd.DateOffset(months=lookahead)
        mask = (letters["upload_date"] >= lo) & (letters["upload_date"] <= hi)
        out[m] = set(letters.loc[mask, "cik"].unique())
    return out


def _match_event(letter_ticker: str, sector: str, event_month: pd.Timestamp,
                 sector_to_tickers: dict[str, list[str]],
                 cik_set_excluded: set[str], ticker_to_cik: dict[str, str],
                 size_proxy: pd.DataFrame,
                 k: int = K_MATCH, band: float = SIZE_BAND_FRAC) -> tuple[list[str], str]:
    """Return (matched_tickers, drop_reason). drop_reason='' on success."""
    # Size proxy is taken at the END of the prior month (= event_month - 1m)
    proxy_month = (event_month - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
    if proxy_month not in size_proxy.index:
        return [], "no_size_month"
    if letter_ticker not in size_proxy.columns:
        return [], "letter_no_size"
    letter_size = size_proxy.at[proxy_month, letter_ticker]
    if pd.isna(letter_size) or letter_size <= 0:
        return [], "letter_size_nan"

    sector_tickers = sector_to_tickers.get(sector, [])
    candidates = []
    lo, hi = letter_size * (1 - band), letter_size * (1 + band)
    row = size_proxy.loc[proxy_month]
    for t in sector_tickers:
        if t == letter_ticker:
            continue
        cik = ticker_to_cik.get(t)
        if cik is None or cik in cik_set_excluded:
            continue
        s = row.get(t, np.nan)
        if pd.isna(s) or s <= 0:
            continue
        if lo <= s <= hi:
            candidates.append((t, abs(s - letter_size) / letter_size))
    if not candidates:
        return [], "no_match"
    # Closest on size, take K
    candidates.sort(key=lambda x: x[1])
    return [t for t, _ in candidates[:k]], ""


def main() -> int:
    print("[day6] loading inputs...")
    ev = pd.read_parquet(EVENTS)
    rets = pd.read_parquet(RETURNS)
    rets["date"] = pd.to_datetime(rets["date"])
    universe = pd.read_parquet(UNIVERSE)
    letters = _load_letter_dates(PAIRS)
    print(f"[day6] events={len(ev)}, R3K returns={len(rets)} rows / {rets['ticker'].nunique()} tickers, "
          f"universe={len(universe)} CIKs, letters={len(letters)}")

    # ---- size proxy table (NB: price-as-size proxy, no shares outstanding) ----
    size_proxy = _build_size_proxy(rets)         # month-end x ticker
    log_pivot = _build_log_ret_pivot(rets)
    mkt_log = _load_french_market()
    print(f"[day6] size proxy = adjusted-close price at end of prior month "
          f"(no volume / no shares outstanding column available)")

    # ---- universe / sector / cik <-> ticker maps ----
    ticker_to_cik = dict(zip(universe["ticker"], universe["cik"]))
    cik_to_ticker = dict(zip(universe["cik"], universe["ticker"]))
    sector_to_tickers: dict[str, list[str]] = (
        universe.groupby("sector")["ticker"].apply(list).to_dict()
    )

    # ---- per-month letter recipient set (rolling 12m back, 3m forward) ----
    all_months = size_proxy.index
    excl = _build_letter_recipients_per_month(letters, all_months)

    # ---- compute matched control BHAR per event ----
    diagnostics = {"events_total": len(ev), "drop_reason": {},
                   "matched_count_distribution": {},
                   "spec": {"K": K_MATCH, "size_band_frac": SIZE_BAND_FRAC,
                            "lookback_months": LOOKBACK_MONTHS,
                            "lookahead_months": LOOKAHEAD_MONTHS,
                            "size_proxy": "price_end_of_prior_month"}}

    parts = []
    for spec in SIGNAL_SPECS:
        leg = spec["leg"]
        kind = spec["ret_kind"]   # 'bhar' or 'car'
        h = spec["horizon"]
        start_col = spec["start_col"]
        # the recipient return col for this signal
        rec_col = f"{kind}_{leg}_{h}m"
        # Per-event matched control return
        rows = []
        drops = {"no_recipient_ret": 0, "no_match": 0, "no_size_month": 0,
                 "letter_no_size": 0, "letter_size_nan": 0,
                 "matched_returns_empty": 0}
        n_matches = []
        for _, row in ev.iterrows():
            recipient_ret = row.get(rec_col)
            if recipient_ret is None or (isinstance(recipient_ret, float) and np.isnan(recipient_ret)):
                drops["no_recipient_ret"] += 1
                continue
            event_month = pd.Timestamp(row[start_col])
            ticker = row["ticker"]
            sector = row["sector"]
            cik = row["cik"]
            # Exclude all letter recipients in the rolling window AND the event firm itself
            excluded = excl.get(event_month + pd.offsets.MonthEnd(0), set()) | {cik}
            matched, why = _match_event(
                ticker, sector, event_month, sector_to_tickers,
                excluded, ticker_to_cik, size_proxy)
            if not matched:
                drops[why if why else "no_match"] = drops.get(why if why else "no_match", 0) + 1
                continue
            # Compute matched-control return
            ctrl_rets = []
            for mt in matched:
                if kind == "bhar":
                    r = _bhar_window(log_pivot, mkt_log, mt, event_month, h)
                else:
                    r = _car_window(log_pivot, mkt_log, mt, event_month, h)
                if r is not None:
                    ctrl_rets.append(r)
            if not ctrl_rets:
                drops["matched_returns_empty"] += 1
                continue
            ctrl_ret = float(np.mean(ctrl_rets))
            n_matches.append(len(ctrl_rets))
            rows.append({
                "month": event_month,
                "sector": sector,
                "ticker": ticker,
                "cik": cik,
                "severity": float(row["upload_severity_mean"]) if not pd.isna(row["upload_severity_mean"]) else 0.0,
                "recipient_ret": float(recipient_ret),
                "control_ret": ctrl_ret,
                "n_controls": len(ctrl_rets),
            })

        per_event = pd.DataFrame(rows)

        # Diagnostics
        diagnostics["drop_reason"][spec["signal_id"]] = drops
        if n_matches:
            diagnostics["matched_count_distribution"][spec["signal_id"]] = {
                "median": int(np.median(n_matches)),
                "mean": float(np.mean(n_matches)),
                "min": int(np.min(n_matches)),
                "max": int(np.max(n_matches)),
                "n_events_kept": len(per_event),
            }

        if per_event.empty:
            print(f"[day6] {spec['signal_id']}: no events kept after match — skipped")
            continue

        # Aggregate by month: severity-weighted short on recipients minus eq-weight long on controls
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
            out_rows.append({
                "month": pd.Timestamp(month),
                "signal_id": spec["signal_id"],
                "horizon_months": h,
                "raw_return": net,
                "short_leg_return": short_ret,
                "long_leg_return": long_ret,
                "n_short": int(len(grp)),
                "n_sectors": int(grp["sector"].nunique()),
            })
        df = pd.DataFrame(out_rows)
        if df.empty:
            print(f"[day6] {spec['signal_id']}: no valid months — skipped")
            continue
        parts.append(df)
        sharpe = df["raw_return"].mean() / df["raw_return"].std() * np.sqrt(12) if df["raw_return"].std() else float("nan")
        print(f"[day6] {spec['signal_id']}: months={len(df)} "
              f"events_kept={len(per_event)}/{len(ev)} "
              f"mean={df['raw_return'].mean():.4f} std={df['raw_return'].std():.4f} sharpe={sharpe:.2f}")

    if not parts:
        print("[day6] no factor returns produced.")
        return 1
    out = pd.concat(parts, ignore_index=True).sort_values(["signal_id", "month"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    OUT_DIAG.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf-8")
    print(f"[day6] wrote {OUT} (rows={len(out)})")
    print(f"[day6] wrote {OUT_DIAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
