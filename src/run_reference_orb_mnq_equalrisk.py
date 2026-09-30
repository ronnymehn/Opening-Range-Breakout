"""MNQ equal-dollar-risk sizing ($2,000/trade, contracts sized from the
trigger-bar close), 1R target, split 2021-2024 (in-sample) vs 2025-2026
(holdout). MNQ's $2/pt granularity keeps trades NQ's $20/pt sizing has to
skip. Note the 2025-2026 holdout has already been looked at several times, so
it is no longer a clean out-of-sample test. Writes to data/results/mnq_equalrisk_*/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_orb  # noqa: E402

PERIODS = {"2021-2024": ("2021-01-01", "2025-01-01"), "2025-2026": ("2025-01-01", "2027-01-01")}

if __name__ == "__main__":
    for label, (start, end) in PERIODS.items():
        run_orb.main(["--instrument", "MNQ", "--start", start, "--end", end, "--targets", "1.0",
                      "--sizing", "orb_risk_target", "--risk-budget", "2000", "--benchmark",
                      "--tag", f"equalrisk_{label}"] + sys.argv[1:])
