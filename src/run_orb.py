"""Command-line ORB backtest: target sweep, robustness variants, benchmarks,
significance statistics and a generated report.

Examples
  python src/run_orb.py --instrument MNQ --start 2021-01-01 --end 2025-01-01 \
      --targets 0.4:2.0:0.1 --slippage-ticks 1,2,3 --robustness --benchmark
  python src/run_orb.py --instrument NQ --targets 1.0 --sizing orb_risk_target --risk-budget 2000

Every configuration in a run counts as a trial. The Deflated Sharpe columns
use the number of configurations in this run and each extra --trial-counts
value (the default 100 approximates every variant tried in this repo so far).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orb_engine as eng  # noqa: E402
import orb_stats as st  # noqa: E402
from fomc_dates import FOMC_DATES, load_event_dates  # noqa: E402

RESULTS_DIR = eng.DATA_DIR / "results"
DEFAULT_FIXED_QTY = {"MNQ": 5, "NQ": 1}
# All-in MNQ cost per side (exchange + clearing + NFA ~ $0.55 plus commission),
# used by the --robustness cost scenario.
ALL_IN_COMMISSION = {"MNQ": 0.85}
DATA_NOTE = ("Continuous volume-rolled (MNQ.V.0) / calendar-rolled (NQ.C.0) futures from QuantPad; "
             "most likely NOT back-adjusted (unverified -- confirm with the data provider). Intraday P&L is "
             "unaffected; %-of-notional columns use raw contract prices; realized vol and ATR ignore returns "
             "within 2 sessions of each quarterly roll.")


def parse_targets(text: str) -> list[float | None]:
    if ":" in text:
        lo, hi, step = (float(v) for v in text.split(":"))
        return [round(v, 4) for v in np.arange(lo, hi + step / 2, step)]
    return [None if v.strip().lower() == "none" else float(v) for v in text.split(",")]


def target_label(target_r: float | None) -> str:
    return "notarget" if target_r is None else f"{target_r:.1f}r"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrument", choices=sorted(eng.INSTRUMENTS), default="MNQ")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-01-01", help="exclusive")
    p.add_argument("--targets", default="0.4:2.0:0.1", help="lo:hi:step or comma list; 'none' = no target")
    p.add_argument("--sizing", choices=eng.SIZINGS, default="fixed")
    p.add_argument("--fixed-qty", type=int, default=None)
    p.add_argument("--risk-budget", type=float, default=1_000.0)
    p.add_argument("--size-from", choices=["trigger_close", "fill"], default="trigger_close")
    p.add_argument("--slippage-ticks", default="1", help="comma list; each value runs the whole target sweep")
    p.add_argument("--stop-slippage-ticks", type=int, default=None)
    p.add_argument("--commission", type=float, default=None, help="$ per contract per side")
    p.add_argument("--time-exit", default="14:00")
    p.add_argument("--last-entry", default=None)
    p.add_argument("--side", choices=["long", "short", "both"], default="long")
    p.add_argument("--stop-mode", choices=["or_opposite", "or_mid", "atr_frac"], default="or_opposite")
    p.add_argument("--stop-atr-frac", type=float, default=0.5)
    p.add_argument("--or-width-atr", default=None, help="lo,hi band for opening-range width / ATR")
    p.add_argument("--exits-on-entry-bar", choices=["both", "stop_only", "none"], default="both")
    p.add_argument("--event-dates", default=None, help="CSV with a `date` column of extra days to skip")
    p.add_argument("--no-fomc-filter", action="store_true")
    p.add_argument("--allow-partial-sessions", action="store_true",
                   help="trade holiday/early-close sessions (exit at the last bar); off by default")
    p.add_argument("--robustness", action="store_true",
                   help="add all-in cost, 3-tick stop slippage, no-target and 15:55-exit variants")
    p.add_argument("--benchmark", action="store_true",
                   help="add always-long 10:05->time-exit and matched no-stop/no-target controls")
    p.add_argument("--trial-counts", default="100", help="extra trial counts N for the Deflated Sharpe")
    p.add_argument("--robust-target", type=float, default=1.0, help="target_r used by the robustness variants")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--no-trade-logs", action="store_true")
    return p


def base_config(args) -> eng.StrategyConfig:
    skip = frozenset() if args.no_fomc_filter else FOMC_DATES
    if args.event_dates:
        skip = skip | load_event_dates(args.event_dates)
    width = tuple(float(v) for v in args.or_width_atr.split(",")) if args.or_width_atr else None
    return eng.StrategyConfig(
        sizing=args.sizing, fixed_qty=args.fixed_qty or DEFAULT_FIXED_QTY[args.instrument],
        risk_budget=args.risk_budget, size_from=args.size_from, stop_slippage_ticks=args.stop_slippage_ticks,
        commission_per_side=args.commission, time_exit=args.time_exit, last_entry=args.last_entry,
        skip_dates=skip, require_full_session=not args.allow_partial_sessions, side=args.side,
        stop_mode=args.stop_mode, stop_atr_frac=args.stop_atr_frac, or_width_atr_range=width,
        exits_active_on_entry_bar=args.exits_on_entry_bar)


def build_configs(args, base: eng.StrategyConfig) -> tuple[dict, list[str]]:
    """{name: config} plus the names forming the primary target sweep (used for PBO)."""
    configs, sweep = {}, []
    slippages = [int(v) for v in args.slippage_ticks.split(",")]
    for slip in slippages:
        for target_r in parse_targets(args.targets):
            name = f"target_{target_label(target_r)}_slip{slip}"
            configs[name] = base.with_(target_r=target_r, slippage_ticks=slip)
            if slip == slippages[0]:
                sweep.append(name)
    if args.robustness:
        robust = base.with_(target_r=args.robust_target, slippage_ticks=slippages[0])
        label = target_label(args.robust_target)
        if args.instrument in ALL_IN_COMMISSION:
            configs[f"target_{label}_allin_cost"] = robust.with_(commission_per_side=ALL_IN_COMMISSION[args.instrument])
        configs[f"target_{label}_stopslip3"] = robust.with_(stop_slippage_ticks=3)
        configs[f"notarget_exit{base.time_exit.replace(':', '')}"] = robust.with_(target_r=None)
        configs["notarget_exit1555"] = robust.with_(target_r=None, time_exit="15:55")
        configs[f"target_{label}_exit1555"] = robust.with_(time_exit="15:55")
    return configs, sweep


def cost_lines(spec: eng.InstrumentSpec, configs: dict) -> list[str]:
    seen = sorted({(c.commission(spec), c.slippage_ticks, c.stop_slip()) for c in configs.values()})
    return [f"  - ${comm:.2f}/contract/side commission, {slip} tick entry/target/time slippage, "
            f"{stop} tick stop slippage" for comm, slip, stop in seen]


def write_report(path: Path, args, spec, base, configs, summary, significance, pbo, n_sessions) -> None:
    lines = [
        f"# {spec.name} 30-minute opening range breakout{' -- ' + args.tag if args.tag else ''}", "",
        "_Generated by src/run_orb.py; do not edit by hand._", "",
        f"- **Sample**: {args.start} to {args.end} (end exclusive), {n_sessions} valid sessions.",
        f"- **Data**: {DATA_NOTE}",
        f"- **Opening range**: {base.or_start} ET, {base.or_bars} five-minute bars. Entry on the next bar's open "
        f"after the first close beyond the range ({base.side}). Stop: {base.stop_mode}. Time exit at the "
        f"{base.time_exit} bar open unless a config says otherwise. Same-bar stop and target: stop first. "
        f"Stops that gap through fill at the bar open.",
        f"- **Sessions**: sessions without a {base.time_exit} bar (holidays, early closes) are "
        f"{'traded to their last bar' if not base.require_full_session else 'excluded'}; "
        f"{len(base.skip_dates)} skip dates (FOMC{' + event file' if args.event_dates else ''}) count as flat days.",
        f"- **Sizing**: {base.sizing}" + (f", ${base.risk_budget:,.0f} risk budget sized from the {base.size_from}"
                                          if base.sizing == "orb_risk_target" else f", base {base.fixed_qty} contracts"),
        "- **Costs** (per config, see names):", *cost_lines(spec, configs),
        f"- **Trials**: {len(configs)} configurations in this run. Deflated Sharpe is shown for several trial "
        "counts; the target_r variants are highly correlated, so the true effective N lies between them.",
        "", "## Results", "", "```", summary.round(2).to_string(), "```",
        "", "## Significance", "",
        "Sharpe CIs are 95% bootstrap intervals (i.i.d. and stationary-block) on daily P&L incl. flat days. "
        "PSR = probability the true Sharpe is above zero; DSR = the same against the best Sharpe expected "
        "from N skill-less trials.", "", "```", significance.round(3).to_string(), "```",
    ]
    if pbo is not None:
        lines += ["", f"**Probability of backtest overfitting (CSCV, 16 blocks) across the target sweep: "
                      f"{pbo:.2f}.** Near 0.5 or above means choosing the best in-sample target_r is no better "
                      "than chance out of sample."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    spec = eng.INSTRUMENTS[args.instrument]
    base = base_config(args)
    configs, sweep = build_configs(args, base)
    out_dir = Path(args.out_dir) if args.out_dir else RESULTS_DIR / f"{spec.name.lower()}_{args.tag or 'run'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = eng.fetch_5m(spec, args.start, args.end)
    annual_vol = eng.realized_volatility(bars, base.vol_lookback) if base.sizing in eng.VOL_SIZINGS else None
    atr = eng.daily_atr(bars, base.atr_lookback) if (base.stop_mode == "atr_frac" or base.or_width_atr_range) else None

    trades_by_config, daily_by_config, summary = {}, {}, {}
    for name, cfg in configs.items():
        trades = eng.simulate(bars, spec, cfg, annual_vol=annual_vol, atr=atr)
        sessions = eng.session_dates(bars, cfg)
        trades_by_config[name] = trades
        daily_by_config[name] = st.daily_pnl(trades, sessions)
        summary[name] = eng.metrics(trades, sessions)
        if not args.no_trade_logs:
            trades.to_csv(out_dir / f"{name}_trades_detail.csv", index=False)
            eng.export_trade_log(trades, out_dir / f"{name}.tvtrades.csv", spec)

    if args.benchmark:
        reference = min(sweep, key=lambda n: abs((configs[n].target_r or 99) - args.robust_target))
        cfg = configs[reference]
        sessions = eng.session_dates(bars, cfg)
        controls = {
            f"benchmark_long_1005_to_{cfg.time_exit.replace(':', '')}": eng.simulate_benchmark(bars, spec, cfg),
            f"control_{reference}_no_stop_no_target": eng.matched_control(trades_by_config[reference], bars, spec, cfg),
        }
        for name, trades in controls.items():
            summary[name] = eng.metrics(trades, sessions)
            daily_by_config[name] = st.daily_pnl(trades, sessions)
            if not args.no_trade_logs:
                trades.to_csv(out_dir / f"{name}_trades_detail.csv", index=False)

    summary_df = pd.DataFrame(summary).T
    summary_df.index.name = "configuration"
    summary_df.to_csv(out_dir / "summary.csv")
    trial_counts = sorted({len(configs), *(int(v) for v in args.trial_counts.split(",") if v)})
    strategy_daily = {n: d for n, d in daily_by_config.items() if n in configs}
    significance = st.significance_table(strategy_daily, trial_counts=trial_counts)
    for name in set(daily_by_config) - set(configs):  # controls: no deflation, they are not trials
        significance = pd.concat([significance, st.significance_table({name: daily_by_config[name]})
                                  .drop(columns=["dsr_N1"])])
    significance.to_csv(out_dir / "significance.csv")
    pbo = None
    if len(sweep) >= 2:
        matrix = pd.concat([daily_by_config[n] for n in sweep], axis=1).fillna(0.0).to_numpy()
        if len(matrix) >= 16:
            pbo = st.pbo_cscv(matrix)
    n_sessions = len(eng.session_dates(bars, base))
    write_report(out_dir / "report.md", args, spec, base, configs, summary_df, significance, pbo, n_sessions)

    pd.set_option("display.width", 220)
    print(summary_df.round(2).to_string())
    print(significance.round(3).to_string())
    if pbo is not None:
        print(f"PBO across target sweep: {pbo:.2f}")
    print(f"\nWrote results to {out_dir}")
    return {"summary": summary_df, "significance": significance, "pbo": pbo, "trades": trades_by_config}


if __name__ == "__main__":
    main()
