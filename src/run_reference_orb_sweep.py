"""MNQ reference ORB target_r sweep, 2021-2024 (5 micros, 0.4R-2.0R), plus
1-3 tick slippage across every target, the --robustness variants (all-in
costs, 3-tick stops, no target, 15:55 exit) and the always-long benchmark.
Thin wrapper around run_orb.py; extra CLI arguments are passed through.
Writes to data/results/mnq_reference_sweep/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_orb  # noqa: E402

ARGS = ["--instrument", "MNQ", "--start", "2021-01-01", "--end", "2025-01-01",
        "--targets", "0.4:2.0:0.1", "--sizing", "fixed", "--fixed-qty", "5",
        "--slippage-ticks", "1,2,3", "--robustness", "--robust-target", "1.4", "--benchmark",
        "--tag", "reference_sweep"]

if __name__ == "__main__":
    run_orb.main(ARGS + sys.argv[1:])
