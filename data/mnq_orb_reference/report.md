# MNQ 30-Minute Opening Range Breakout

> **STALE -- kept for history.** These results came from the pre-refactor engine and include known
> biases: 13-15 holiday/early-close trades per config whose "14:00" exit filled at the 18:00 reopen,
> early-close trades that were silently dropped (only when the target was missed), and 18 FOMC-day
> trades (the FOMC filter was added later). The data is most likely an unadjusted continuous series,
> not back-adjusted as stated below. The ORB-risk, Harvey, Moreira-Muir and 16/24-session variants
> described below are not in this table. After the corrections, no configuration has a t-stat above
> 2 and PBO across the target sweep is 0.77 -- see `data/analysis/offline_significance.md`.
> Re-run with `python src/run_reference_orb_sweep.py` (writes to `data/results/`).

- **Sample**: 2021-01-01 through 2025-01-01 (end exclusive).
- **Instrument/data**: MNQ volume-rolled continuous futures, 5-minute bars, back-adjusted.
- **Entry**: First completed 5-minute close above the 09:30-10:00 ET opening-range high; fill at next bar open.
- **Exit**: Opening-range low stop, target-R sweep, or 14:00 ET open; a same-bar stop/target collision uses stop-first.
- **Costs**: $0.62/contract/side plus 1 tick adverse slippage per side; stress test doubles slippage.
- **Baseline sizing**: 5 MNQ contracts. The ORB-risk variant floors whole MNQ contracts to a $1,000 per-trade risk budget (1% of a $100,000 account).
- **Equal-risk calculation**: contracts = floor($1,000 / [((entry − stop) + exit slippage) × $2 + round-trip commissions]). Quantities are whole MNQ contracts, so modeled loss never exceeds $1,000; a trade is skipped if even one MNQ exceeds the budget.
- **Harvey et al. overlay**: 20-session prior-close realized volatility; contracts = floor(5 × 20% / prior annualized vol), clipped to 1–10 MNQ.
- **Moreira–Muir overlay**: same input and clip, but uses the inverse-variance form: floor(5 × (20% / prior annualized vol)^2).
- **Sensitivity**: the Harvey overlay uses 16- and 24-session lookbacks around the 20-session specification; neither is selected for performance.
- **Validation**: Historical in-sample target-R comparison; 18 configurations were evaluated, so no result is a live-performance estimate.

## Results

```
                                     trades   net_pnl  win_rate_pct  profit_factor  expectancy  avg_win  avg_loss  max_drawdown  sharpe  avg_contracts  median_contracts  avg_planned_risk  min_planned_risk  max_planned_risk  target_exits  stop_exits  time_exits
configuration                                                                                                                                                                                                                                                       
fixed_5_mnq_target_0.4r               612.0  10301.10         66.01           1.07       16.83   398.68   -724.83     -18118.10    0.32            5.0               5.0           1106.08             128.7            5341.2         360.0       103.0       149.0
fixed_5_mnq_target_0.5r               611.0  13439.30         62.19           1.08       22.00   467.44   -710.78     -14624.80    0.39            5.0               5.0           1105.55             128.7            5341.2         315.0       110.0       186.0
fixed_5_mnq_target_0.6r               611.0  24026.30         60.23           1.14       39.32   531.54   -706.10     -13647.50    0.66            5.0               5.0           1105.55             128.7            5341.2         273.0       116.0       222.0
fixed_5_mnq_target_0.7r               611.0  22049.30         58.10           1.12       36.09   564.31   -696.41     -12918.70    0.59            5.0               5.0           1105.55             128.7            5341.2         229.0       119.0       263.0
fixed_5_mnq_target_0.8r               611.0  24402.30         56.46           1.13       39.94   599.70   -686.07     -12407.20    0.63            5.0               5.0           1105.55             128.7            5341.2         181.0       122.0       308.0
fixed_5_mnq_target_0.9r               611.0  28693.55         55.81           1.16       46.96   625.66   -683.91     -13118.45    0.72            5.0               5.0           1105.55             128.7            5341.2         153.0       124.0       334.0
fixed_5_mnq_target_1.0r               610.0  31240.50         55.25           1.17       51.21   644.92   -681.68     -13026.70    0.77            5.0               5.0           1105.23             128.7            5341.2         119.0       126.0       365.0
fixed_5_mnq_target_1.1r               610.0  28917.25         54.75           1.15       47.41   652.11   -684.37     -13829.25    0.71            5.0               5.0           1105.23             128.7            5341.2          95.0       127.0       388.0
fixed_5_mnq_target_1.2r               610.0  30586.00         54.10           1.16       50.14   668.22   -678.31     -14680.00    0.73            5.0               5.0           1105.23             128.7            5341.2          81.0       128.0       401.0
fixed_5_mnq_target_1.3r               610.0  33469.25         53.93           1.18       54.87   680.47   -677.60     -14458.75    0.80            5.0               5.0           1105.23             128.7            5341.2          67.0       128.0       415.0
fixed_5_mnq_target_1.4r               610.0  33790.50         53.93           1.18       55.39   681.44   -677.60     -14237.50    0.80            5.0               5.0           1105.23             128.7            5341.2          52.0       128.0       430.0
fixed_5_mnq_target_1.5r               610.0  31956.75         53.77           1.17       52.39   680.61   -678.31     -14813.75    0.75            5.0               5.0           1105.23             128.7            5341.2          42.0       129.0       439.0
fixed_5_mnq_target_1.6r               610.0  31242.00         53.77           1.16       51.22   678.43   -678.31     -14722.00    0.74            5.0               5.0           1105.23             128.7            5341.2          36.0       129.0       445.0
fixed_5_mnq_target_1.7r               610.0  33508.00         53.77           1.18       54.93   685.34   -678.31     -14630.25    0.78            5.0               5.0           1105.23             128.7            5341.2          32.0       129.0       449.0
fixed_5_mnq_target_1.8r               610.0  31531.00         53.44           1.16       51.69   685.11   -675.40     -14538.50    0.74            5.0               5.0           1105.23             128.7            5341.2          26.0       129.0       455.0
fixed_5_mnq_target_1.9r               610.0  30071.00         53.28           1.16       49.30   683.32   -673.72     -15187.50    0.70            5.0               5.0           1105.23             128.7            5341.2          20.0       129.0       461.0
fixed_5_mnq_target_2.0r               610.0  31530.50         53.28           1.16       51.69   687.82   -673.72     -15187.50    0.73            5.0               5.0           1105.23             128.7            5341.2          18.0       129.0       463.0
fixed_5_mnq_target_1.4r_2x_slippage   610.0  31002.50         53.44           1.16       50.82   683.50   -675.42     -14475.50    0.73            5.0               5.0           1110.23             133.7            5346.2          51.0       128.0       431.0
```