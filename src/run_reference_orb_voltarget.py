"""Same MNQ reference sweep (2021-2024) with the volatility-targeting sizing
schemes instead of fixed 5 micros: Harvey et al. (5 x 20% / vol) and
Moreira-Muir (5 x (20% / vol)^2), clipped to 1-10 MNQ, 20-session prior-close
realized vol (roll days excluded). Writes to data/results/mnq_voltarget_*/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_orb  # noqa: E402

if __name__ == "__main__":
    for sizing in ("harvey_target_vol", "moreira_muir"):
        run_orb.main(["--instrument", "MNQ", "--start", "2021-01-01", "--end", "2025-01-01",
                      "--targets", "0.4:2.0:0.1", "--sizing", sizing, "--fixed-qty", "5",
                      "--tag", f"voltarget_{sizing}"] + sys.argv[1:])
