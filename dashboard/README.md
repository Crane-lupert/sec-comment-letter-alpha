# SEC Comment Letter Alpha — Dashboard

Streamlit app surfacing the Day 4-7 cross-section results.

## Run locally

```bash
uv pip install streamlit plotly  # one-time
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Opens on `http://localhost:8501`.

## Tabs

- **Overview** — headline numbers (Sharpe, alpha, t, p, DSR per Signal A/B × FULL/IS/OOS) with toggle between Day 4 sector-mean control and Day 6 matched control.
- **Topic heatmap** — 14 topic enum × 4 severity bands, color = mean recipient BHAR 2m. Counts table below.
- **Severity distribution** — per-band event count + mean BHAR 2m bar chart.
- **Cumulative returns** — Signal A vs B vs robustness horizons. Cumulative log return.
- **Robustness** (Day 7) — sector / size_quintile / liquidity_quintile stratified alpha.
- **TC sensitivity** (Day 6) — post-cost Sharpe at 5/10/20 bps/month + break-even TC.
- **About** — links to README + limitations.md.

## Data dependencies

The app expects these (gitignored) files to exist; they are produced by:

| File | Producer |
|---|---|
| `data/day4_pairs.jsonl` | `scripts/day4_build_pairs.py` |
| `data/day4_events.parquet` | `scripts/day4_build_panel.py` |
| `data/day4_factor_returns.parquet` | `scripts/day4_construct_signal.py` |
| `data/day4_alpha_summary.json` | `scripts/day4_orthogonalize.py` |
| `data/day6_factor_returns_matched.parquet` | `scripts/day6_signal_matched.py` |
| `data/day6_alpha_summary_matched.json` | `scripts/day4_orthogonalize.py --input ... --output ...` |
| `data/day6_post_tc_summary.json` | `scripts/day6_apply_tc.py` |
| `data/day7_robustness_summary.json` | `scripts/day7_robustness.py` |

Or just run `.venv/Scripts/python.exe scripts/day4_run_all.py` then the day6/day7 scripts in order.

## Deployment (Day 10)

Free options:
- Streamlit Community Cloud (link to GitHub repo, public)
- Hugging Face Spaces

For deployment, copy the data/ files needed by the app into a `dashboard/data/` directory committed to the repo (don't push the full sec-data/ cache).
