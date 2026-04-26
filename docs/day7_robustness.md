# Day 7 robustness — sector / size / liquidity strata

Source: data/day4_events.parquet (697 events with yfinance forward returns).
Min events per stratum: 15; min months: 12.

## Per-stratum alpha (BHAR 2m, severity-weighted long-short with sector-mean control)

| stratum | n_events | n_months | Sharpe | α/yr | t | p |
|---|---|---|---|---|---|---|
| FULL | 691 | 119 | +0.72 | +7.20% | +2.76 | 0.006 |
| sector=Industrials | 195 | 57 | -0.17 | -0.46% | -0.05 | 0.957 |
| sector=Health Care | 48 | — | — | — | — | SKIP |
| sector=Information Technology | 68 | — | — | — | — | SKIP |
| sector=Materials | 71 | 15 | +0.95 | +4.19% | +1.21 | 0.225 |
| sector=Real Estate | 23 | — | — | — | — | SKIP |
| sector=Energy | 57 | — | — | — | — | SKIP |
| sector=Utilities | 9 | — | — | — | — | SKIP |
| sector=Financials | 65 | 15 | +0.84 | +10.59% | +0.99 | 0.320 |
| sector=Consumer Staples | 63 | — | — | — | — | SKIP |
| sector=Consumer Discretionary | 72 | 12 | +1.14 | -1.76% | -0.54 | 0.591 |
| sector=Communication | 20 | — | — | — | — | SKIP |
| size_q0 | 73 | 13 | -1.84 | -96.08% | -2.20 | 0.028 |
| size_q1 | 111 | 26 | +1.14 | +57.13% | +3.71 | 0.000 |
| size_q2 | 146 | 35 | +0.71 | +14.23% | +2.09 | 0.036 |
| size_q3 | 189 | 54 | +0.23 | +0.93% | +0.25 | 0.803 |
| size_q4 | 168 | 40 | +0.00 | -2.14% | -0.60 | 0.548 |
| liq_q0 | 160 | 48 | +0.19 | +2.47% | +0.60 | 0.549 |
| liq_q1 | 148 | 38 | +0.51 | +9.19% | +1.30 | 0.195 |
| liq_q2 | 155 | 47 | -0.09 | +0.19% | +0.03 | 0.976 |
| liq_q3 | 132 | 39 | -0.95 | -6.54% | -1.44 | 0.151 |
| liq_q4 | 90 | 21 | +1.08 | +5.78% | +1.78 | 0.074 |

**Caveat**: this script uses the Day 4 sector-mean-of-recipients control (not the Day 6 matched control). For comprehensive Day 7 rigor, swap the stratum_factor's long-leg construction to use scripts/day6_signal_matched.py matching logic. Done as Day 7 follow-up after Day 6 sample-expansion finishes.