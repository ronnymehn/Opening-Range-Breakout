# Opening Range Breakout (NQ / MNQ)

Backtest research for a 30-minute opening range breakout on Nasdaq-100 futures,
plus a MultiCharts/EasyLanguage version of the same rules.

**Strategy (defaults):** the opening range is 09:30–10:00 ET (six 5-minute bars).
The first 5-minute close above the range high enters long at the next bar's open.
The stop is the range low and the target is `target_r` × risk. A time exit at the
14:00 bar's open applies if neither level is hit. One trade per day, and FOMC days
are skipped. Exchange holidays and early-close sessions (no 14:00 bar) are not traded.

## Current evidence

Run `python src/analyze_trades.py` to reproduce this from the committed trade logs.
It removes the holiday and FOMC trades the old engine included. See
[`data/analysis/offline_significance.md`](data/analysis/offline_significance.md).

- **2021–2024 target sweep:** Sharpe is about 0.3–0.9 on both MNQ and NQ, and no
  target reaches a t-stat of 2. Every 95% bootstrap Sharpe interval includes zero.
- **Choice of target:** the probability of backtest overfitting (PBO) across the
  0.4R–2.0R sweep is **0.77**. Picking the best in-sample target does worse than
  chance out of sample, so don't read the 1.3–1.4R peak as meaningful.
- **2025–2026:** the numbers are stronger (MNQ equal-risk Sharpe about 1.4, CI [0.0, 2.8]).
  But that period has already been examined several times, so it isn't a clean
  holdout. The only untouched data is what arrives after 2026-09.

## Layout

| Path | Purpose |
|---|---|
| `src/orb_engine.py` | Engine. Instrument specs (MNQ/NQ), `StrategyConfig`, data cache, `simulate`, benchmarks, metrics |
| `src/orb_stats.py` | t-stat, bootstrap CIs, Probabilistic / Deflated Sharpe, PBO via CSCV |
| `src/run_orb.py` | Command-line interface: sweeps, robustness variants, benchmarks, generated `report.md` |
| `src/run_reference_orb_*.py` | The original experiments as thin wrappers around `run_orb.py` |
| `src/analyze_trades.py` | Offline re-analysis of the committed trade logs (no market data needed) |
| `src/fomc_dates.py` | FOMC dates 2020–2026 and a loader for your own skip-date CSVs |
| `easylanguage/orb_breakout_multicharts.txt` | MultiCharts version. Read its header for the required chart setup |
| `data/mnq_orb_reference/`, `data/orb_reference_nq/` | **Stale** results from the pre-refactor engine, kept for history |
| `data/results/` | Output of new runs |
| `tests/` | pytest suite on synthetic bars, including golden equivalence with the old engine |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pytest
```

Downloading bars needs the private `quantpad_data` package. 1-minute bars are cached
in `data/<symbol>_1m.parquet`, and a `.meta.json` sidecar records the date range covered.
Later runs only fetch dates that aren't cached yet, and end dates in the future are
clamped to today.

**Data caveat:** `MNQ.V.0` (volume roll) and `NQ.C.0` (calendar roll) are continuous
series that are most likely **not back-adjusted**; confirm this with the data provider.
Intraday P&L is unaffected. Realized volatility and ATR ignore returns within 2
sessions of each quarterly roll.

## Running

```bash
# the reference MNQ sweep + slippage 1-3 ticks + robustness variants + benchmarks
python src/run_reference_orb_sweep.py

# anything else, e.g. both directions, no target, exit near the close
python src/run_orb.py --instrument MNQ --targets none,1.0 --side both --time-exit 15:55 --benchmark --tag both_sides
python src/run_orb.py --help
```

Each run writes the trade logs, `summary.csv`, `significance.csv` and a generated
`report.md` into `data/results/<instrument>_<tag>/`.

**Opt-in options** (all off by default):
- `--side short|both`
- `--stop-mode or_mid|atr_frac`
- `--or-width-atr lo,hi` (filter on range width / ATR)
- `--time-exit` / `--last-entry`
- `--event-dates file.csv` (extra skip days, e.g. a verified list of 10:00 ET release dates)
- `--exits-on-entry-bar`
- `--stop-slippage-ticks`
- `--commission`

**Controls added by `--benchmark`:**
- An always-long 10:05→time-exit trade every session. If the ORB can't beat it, the
  breakout adds nothing beyond intraday drift.
- A matched control: the same entries held to the time exit with no stop or target.

## Checking MultiCharts against the backtest, and demo fills

1. In MultiCharts, export the chart's 1-minute (or 5-minute) MNQ bars to CSV. MultiCharts
   stamps each bar with its close time, and the export should be in Eastern time.
2. Run the Python backtest on those exact bars. QuantPad isn't needed for this:
   ```bash
   python src/run_orb.py --bars-csv mnq_export.csv --start 2026-06-01 --end 2026-10-01 \
       --targets 1.0 --fixed-qty 1 --tag parity
   ```
3. Export the strategy's List of Trades from MultiCharts to CSV (or keep a demo fill log with
   `entry_time,entry_price,exit_time,exit_price,qty,exit_reason`) and compare:
   ```bash
   python src/compare_trades.py --python data/results/mnq_parity/target_1.0r_slip1_trades_detail.csv \
       --other mc_trades.csv --point-value 2
   ```
   Each session is reported as match, mismatch (with the reason), missing or extra. The
   script also prints the typical timestamp offset between platforms and the slippage in
   ticks versus the backtest.

**Trials:** every configuration you try is a trial. The report deflates Sharpe for
the number of configs in the run and for `--trial-counts` (default 100, a rough
count of the variants tried in this repo so far). Decide on new variants in advance
rather than sweeping until something looks good.
