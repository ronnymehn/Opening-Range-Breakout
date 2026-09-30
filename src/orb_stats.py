"""Significance and overfitting statistics for the ORB backtests (numpy and
the standard library only).

Conventions (Bailey & Lopez de Prado): every Sharpe ratio here is PER PERIOD
(daily, not annualized), computed on daily P&L that includes flat days, with
T = number of sessions, skew and RAW (non-excess) kurtosis of that series.
Multiply by sqrt(252) only for display.

References
  - Probabilistic / Deflated Sharpe: Bailey & Lopez de Prado (2014),
    https://papers.ssrn.com/abstract=2460551
  - PBO via CSCV: Bailey, Borwein, Lopez de Prado & Zhu (2016),
    https://papers.ssrn.com/abstract=2326253
  - Stationary bootstrap: Politis & Romano (1994).
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def daily_pnl(trades: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Net P&L per session, zero on sessions without a trade."""
    if trades.empty:
        return pd.Series(0.0, index=sessions)
    return trades.groupby("session_date")["net_pnl"].sum().reindex(sessions, fill_value=0.0)


def sharpe(x) -> float:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def moments(x) -> tuple[float, float]:
    """(skew, raw kurtosis); a normal distribution gives (0, 3)."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean()) / x.std(ddof=0)
    return float((z ** 3).mean()), float((z ** 4).mean())


def sharpe_t_stat(daily) -> tuple[float, float]:
    """(annualized Sharpe, t-statistic of the mean daily P&L)."""
    sr = sharpe(daily)
    return sr * math.sqrt(252), sr * math.sqrt(len(daily))


def probabilistic_sharpe(sr: float, T: int, skew: float, kurt: float, sr0: float = 0.0) -> float:
    """P(true per-period Sharpe > sr0) given an observed per-period `sr`."""
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0 or T < 2:
        return float("nan")
    return _N.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """SR0: the Sharpe the best of `n_trials` skill-less strategies would show."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_trials) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                    + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float, n_trials: int, var_trials: float) -> float:
    """PSR against the expected maximum Sharpe of `n_trials` trials whose
    per-period Sharpes have variance `var_trials`."""
    return probabilistic_sharpe(sr, T, skew, kurt, expected_max_sharpe(n_trials, var_trials))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) resample indices; blocks have geometric length with the
    given mean, wrapping around the end. mean_block=1 is the i.i.d. bootstrap."""
    starts = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < 1.0 / mean_block
    restart[:, 0] = True
    pos = np.arange(n)
    block_start = np.maximum.accumulate(np.where(restart, pos, 0), axis=1)
    first = np.take_along_axis(starts, block_start, axis=1)
    return (first + pos - block_start) % n


def bootstrap_ci(daily, stat: str = "sharpe", n_boot: int = 5_000, mean_block: float = 1.0,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for the mean daily P&L ("mean") or the ANNUALIZED Sharpe ("sharpe")."""
    x = np.asarray(daily, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for chunk in range(0, n_boot, 1_000):
        idx = stationary_bootstrap_indices(len(x), min(1_000, n_boot - chunk), mean_block, rng)
        sample = x[idx]
        if stat == "mean":
            values.append(sample.mean(axis=1))
        else:
            sd = sample.std(axis=1, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                values.append(np.where(sd > 0, sample.mean(axis=1) / sd, np.nan) * math.sqrt(252))
    values = np.concatenate(values)
    lo, hi = np.nanquantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def pbo_cscv(daily_matrix, n_splits: int = 16) -> float:
    """Probability of backtest overfitting via combinatorially symmetric
    cross-validation. `daily_matrix` is (T sessions x N configurations). For
    every way of choosing half the time blocks as in-sample, pick the best
    in-sample configuration and record its out-of-sample rank; PBO is the
    share of splits where that pick lands in the bottom half out-of-sample."""
    m = np.asarray(daily_matrix, dtype=float)
    T, N = m.shape
    if N < 2 or n_splits % 2 or T < n_splits:
        raise ValueError("need >= 2 configurations, an even n_splits and T >= n_splits")
    rows = T - T % n_splits
    blocks = m[:rows].reshape(n_splits, rows // n_splits, N)
    block_sum, block_sq, block_n = blocks.sum(axis=1), (blocks ** 2).sum(axis=1), rows // n_splits

    def block_sharpe(ids):
        n = block_n * len(ids)
        s, q = block_sum[list(ids)].sum(axis=0), block_sq[list(ids)].sum(axis=0)
        var = (q - s ** 2 / n) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(var > 0, (s / n) / np.sqrt(var), -np.inf)

    all_ids = set(range(n_splits))
    below = total = 0
    for is_ids in itertools.combinations(range(n_splits), n_splits // 2):
        best = int(np.argmax(block_sharpe(is_ids)))
        oos = block_sharpe(sorted(all_ids - set(is_ids)))
        rank = (oos < oos[best]).sum() + 0.5 * ((oos == oos[best]).sum() - 1) + 1  # 1..N, ties averaged
        w = rank / (N + 1)
        below += math.log(w / (1 - w)) <= 0
        total += 1
    return below / total


def significance_table(daily_by_config: dict[str, pd.Series], trial_counts=(1,),
                       n_boot: int = 5_000, mean_block: float = 5.0) -> pd.DataFrame:
    """Per-configuration Sharpe, t-stat, bootstrap CIs (i.i.d. and stationary
    block), PSR against zero and DSR for each assumed number of trials. Trial
    Sharpe variance comes from the configurations passed in, so with fewer
    than two configurations DSR is left blank rather than reported as PSR."""
    srs = {name: sharpe(d) for name, d in daily_by_config.items()}
    finite = [v for v in srs.values() if np.isfinite(v)]
    var_trials = float(np.var(finite, ddof=1)) if len(finite) > 1 else float("nan")
    rows = {}
    for name, daily in daily_by_config.items():
        sr, T = srs[name], len(daily)
        skew, kurt = moments(daily) if np.nanstd(daily) > 0 else (float("nan"), float("nan"))
        ann, t_stat = sharpe_t_stat(daily)
        row = {"sessions": T, "sharpe_ann": ann, "t_stat": t_stat}
        row["sharpe_ci_iid"] = "[{:.2f}, {:.2f}]".format(*bootstrap_ci(daily, n_boot=n_boot))
        row[f"sharpe_ci_block{mean_block:g}"] = "[{:.2f}, {:.2f}]".format(
            *bootstrap_ci(daily, n_boot=n_boot, mean_block=mean_block))
        row["psr_vs_0"] = probabilistic_sharpe(sr, T, skew, kurt)
        for n in trial_counts:
            row[f"dsr_N{n}"] = (deflated_sharpe(sr, T, skew, kurt, n, var_trials)
                                if np.isfinite(var_trials) else float("nan"))
        rows[name] = row
    table = pd.DataFrame(rows).T
    table.index.name = "configuration"
    return table
