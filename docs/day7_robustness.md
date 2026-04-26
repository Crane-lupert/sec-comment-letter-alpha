# Day 7 robustness — sector / size / liquidity strata

Source: data/day4_events.parquet (697 events with yfinance forward returns).
Min events per stratum: 15; min months: 12.

## Per-stratum alpha (BHAR 2m, severity-weighted long-short with sector-mean control)

| stratum | n_events | n_months | Sharpe | α/yr | t | p |
|---|---|---|---|---|---|---|
| FULL | 770 | 124 | +0.68 | +6.56% | +2.25 | 0.025 |
| sector=Industrials | 223 | 66 | -0.29 | -4.79% | -0.50 | 0.614 |
| sector=Health Care | 51 | — | — | — | — | SKIP |
| sector=Information Technology | 76 | 13 | +0.07 | +0.63% | +0.33 | 0.742 |
| sector=Materials | 80 | 15 | +0.95 | +4.19% | +1.21 | 0.225 |
| sector=Real Estate | 25 | — | — | — | — | SKIP |
| sector=Energy | 63 | 13 | -0.09 | -13.49% | -1.42 | 0.155 |
| sector=Utilities | 15 | — | — | — | — | SKIP |
| sector=Financials | 69 | 15 | +0.84 | +10.59% | +0.99 | 0.320 |
| sector=Consumer Staples | 63 | — | — | — | — | SKIP |
| sector=Consumer Discretionary | 84 | 17 | -0.76 | -12.08% | -4.97 | 0.000 |
| sector=Communication | 21 | — | — | — | — | SKIP |
| size_q0 | 83 | 15 | -1.50 | -84.32% | -2.06 | 0.040 |
| size_q1 | 127 | 32 | +1.11 | +26.28% | +3.15 | 0.002 |
| size_q2 | 160 | 42 | +0.63 | +9.21% | +1.61 | 0.108 |
| size_q3 | 208 | 62 | -0.19 | -3.92% | -0.78 | 0.438 |
| size_q4 | 188 | 48 | -0.23 | -2.40% | -0.74 | 0.459 |
| liq_q0 | 169 | 50 | +0.00 | +1.24% | +0.34 | 0.733 |
| liq_q1 | 171 | 47 | +0.55 | +7.83% | +1.42 | 0.156 |
| liq_q2 | 177 | 52 | -0.25 | -2.75% | -0.49 | 0.623 |
| liq_q3 | 147 | 43 | -1.03 | -5.63% | -1.20 | 0.232 |
| liq_q4 | 100 | 22 | +1.22 | +8.45% | +1.53 | 0.126 |

**Caveat**: this script uses the Day 4 sector-mean-of-recipients control (not the Day 6 matched control). For comprehensive Day 7 rigor, swap the stratum_factor's long-leg construction to use scripts/day6_signal_matched.py matching logic. Done as Day 7 follow-up after Day 6 sample-expansion finishes.