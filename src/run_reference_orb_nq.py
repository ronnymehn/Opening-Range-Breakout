"""Reference ORB target_r sweep on NQ (full-size E-mini, $20/pt, $2.30/side),
2020-2026. 1 NQ ~= 10 micros, so fixed sizing is about 2x the notional of the
"5 micros" MNQ runs -- not a risk-equivalent comparison. Thin wrapper around
run_orb.py keeping this script's original options. Writes to
data/results/nq_<tag>/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_orb  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default="2027-01-01")
    p.add_argument("--out-suffix", default="")
    p.add_argument("--sizing", default="fixed_1", choices=["fixed_1", "equal_risk"])
    p.add_argument("--target-r", type=float, default=None, help="run a single target_r instead of the full sweep")
    args = p.parse_args()
    sizing = ["--sizing", "fixed", "--fixed-qty", "1"] if args.sizing == "fixed_1" else \
        ["--sizing", "orb_risk_target", "--risk-budget", "2000"]
    targets = str(args.target_r) if args.target_r is not None else "0.4:2.0:0.1"
    run_orb.main(["--instrument", "NQ", "--start", args.start, "--end", args.end, "--targets", targets,
                  *sizing, "--tag", f"reference_{args.sizing}{args.out_suffix}"])
