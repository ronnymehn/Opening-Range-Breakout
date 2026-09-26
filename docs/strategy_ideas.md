# VIX-expected-move strategy roadmap

Base data/pipeline (NQ 1m bars + VIX daily closes, 2019-12 -> present) is
already built in src/fetch_data.py. src/strategy.py + src/backtest.py run the
original band-breakout/momentum strategy (see README-less history in chat --
tuned stop/sizing attempts on 2020-2024 in-sample failed the 2025-2026
holdout; fixed 1 NQ contract, stop_r=1.0, target_r=1.0, vix_cap=None remains
the best-validated config).

Discipline for all of these: tune only on 2021-2024 (2020's COVID crash is a
single outlier event, excluded from tuning), then check once, unmodified, on
the 2025-2026 holdout. Don't re-tune on the holdout.

## 1. Fade-the-band mean reversion -- TESTED, NEGATIVE RESULT
Mirror image of the breakout strategy: when price closes beyond the
VIX-implied band, fade back toward the open instead of following through.
Rationale: VIX tends to overstate subsequent realized vol (the variance risk
premium), so the band breach may often be an overreaction relative to what
the day actually needed to move. Implemented as `mode="fade"` in
strategy.simulate_session (alongside the existing `mode="breakout"`), reusing
the same backtest.py pipeline/costs/sizing. See src/tune_fade.py.

Result: every target_r x stop_r combo in a 20-point grid (target_r in
0.25-1.0, stop_r in 0.5-2.0) was net-negative on 2021-2024 in-sample --
Sharpe ranged -1.07 to -1.90, no positive-edge config found. Not a
calibration problem: entering a fade exactly at the moment of a band breach
means entering directly into active momentum, the worst possible timing for
a mean-reversion trade in a market with real directional persistence (NQ had
this, esp. 2022). Did not proceed to the 2025-2026 holdout since nothing
cleared the in-sample bar. Conclusion: the "VIX overstates realized vol"
premise may still be real, but an immediate breach-triggered futures fade is
the wrong way to operationalize it -- a delayed/confirmed reversal entry
(#3) or direct options premium selling (#4) are more promising next steps
than more breach-triggered-fade tuning.

## 2. Relative/dynamic VIX regime filter (overlay on the breakout strategy)
The absolute VIX-level cutoff tried earlier (VIX > 35) looked great on
2020-2024 in-sample but failed the 2025-2026 holdout -- "35" means something
different in different VIX regimes. Instead, filter on VIX's position
relative to its own recent history (e.g. percentile vs trailing 20-60 day
range, or rising vs falling) so the cutoff adapts across years instead of
being a fixed number. Only take breakout trades when VIX is genuinely
expanding; skip/fade when it's elevated-but-flat.

## 3. Implied-vs-realized divergence signal -- TESTED, NEGATIVE RESULT
Track how much of the previous day's actual NQ move consumed of its
VIX-predicted expected move (a realized/implied ratio). Persistent
undershoot of the prediction may justify fading more confidently; overshoot
may signal a genuine regime shift worth trusting for continuation. A
self-adjusting version of #2, built from the same NQ/VIX inputs. Implemented
in src/divergence.py (trailing mean of realized_range/em_points, shifted to
avoid lookahead) + backtest.run_backtest_switching (per-session mode chosen
from the signal, default_mode="breakout" so it reduces to the baseline
outside the fade regime). Swept lookback in [5,10,20,40] x low_thresh in
[0.6-0.9] x high_thresh in [1.1-1.4] on 2021-2024, see src/tune_divergence.py.

Result: best in-sample config (lookback=10, low_thresh=0.9, only 20/429
trades switched to fade) looked excellent -- net +40% ($82,188->$114,743),
Sharpe +40% (1.44->2.01) vs the pure-breakout baseline. Validated on
2025-2026 holdout unmodified: net $29,213 vs baseline's $35,236, max_dd
-$50,012 vs baseline's -$45,987, Sharpe 0.88 vs baseline's 1.06 -- WORSE than
doing nothing on every metric. Same failure signature as the earlier
stop_r/vix_cap tuning and vol-normalized sizing attempts: big in-sample gain
from a small affected subgroup (20-21 trades) that doesn't generalize.

Three independent refinement attempts (stop/vix_cap tuning, vol-normalized
sizing, divergence switching) have now all shown this identical pattern.
Read as a meta-finding: with only ~429 in-sample trades, there isn't enough
data for threshold/lookback-style parameter search to reliably find real
structure rather than in-sample noise on this dataset. Before trying more
filters/signals in this same style, consider whether #4 (options premium
harvesting, a structurally different approach rather than another parameter
search) is more likely to add real value.

## 4. Direct options premium harvesting -- STOPPED, NEGATIVE SO FAR (partial sample)
Rather than approximating the VIX risk premium through NQ futures direction,
sell it directly: QQQ iron condor, short call/put at the VIX-implied band
strikes, long wings $2 further out for defined risk, sized off the nearest
available expiration (QQQ has near-daily expiries from ~2021 on), entered at
the day's open print and closed at the same day's close print. Implemented
in src/options_data.py + src/iron_condor.py + src/run_iron_condor.py
(includes resume support -- see checkpoint/progress-log mechanism in
iron_condor.run_iron_condor_backtest).

Data pull is much slower than the futures pipeline (~10-50 sec/session,
option chain + 4 per-contract price lookups per day vs one bulk yearly pull),
so this ran as a multi-hour background job rather than an interactive one.

First pass hit a real bug: fetch_qqq_daily() tz_converted an
already-UTC-midnight-anchored daily bar timestamp, shifting every session's
date back by one calendar day. The QQQ open price stayed correct for its
true session, but the option chain/price data was fetched for the WRONG
(previous) day throughout -- invalidating that run's results (which had
looked implausibly good: 87-90% win rate, PF 11-14, Sharpe ~12, essentially
a smooth line up). Killed and restarted after fixing the date handling and
verifying it against known-true QQQ opens.

Corrected results (2021-01-01 through 2021-12-16, 239 trades, stopped by
user before finishing the full 2021-2024 window): win rate 57.7%, net
**-$1,042**, profit factor 0.69, max drawdown -$993, Sharpe **-1.69**. A
complete reversal from the buggy run -- once priced against the correct day,
this specific iron condor structure loses money over the sample tested.
Notably win rate stayed >50% but profit factor was still <1, meaning losing
trades outweighed winners in size -- consistent with a defined-risk
short-premium structure where the occasional max-loss (~wing_width - credit)
trade costs more than a typical small win collects.

Stopped mid-run (partial 2021 only, never reached 2022-2024 or a 2025-2026
holdout check) once the negative trend was clear across ~240 trades. Not
proof the underlying "VIX overprices realized vol" premise is wrong, but
this specific instantiation (entry exactly at the band strikes, $2 wings,
1-day hold, no spread cost modeled) doesn't show it. A narrower wing width,
different strike selection (e.g. sell further OTM for a smaller/safer
credit), or actually modeling bid/ask spread costs (which would make this
result WORSE, not better, since none are currently modeled) are the more
likely next steps if this is revisited -- but not attempted here.

## New ideas (generated after closing out #1-4), each targeting a specific
## observed pattern rather than repeating the same overfitting mistake

### 5. Initial Balance breakout + VIX band as TARGET only -- TESTED, NEGATIVE RESULT
Every strategy tried so far used the VIX band as the ENTRY trigger, despite
the original idea's own caveat that it's "a sanity check on move size, not a
standalone signal." Decoupled that: trigger on the breakout of the first
30-60min opening range (a session-structure signal independent of VIX/
em_points), stop at the opposite IB bound (the classic IB-failure level, not
a fitted parameter), target = entry +/- target_r * em_points (VIX-sized).
Implemented as strategy.simulate_session_ib + backtest.run_backtest_ib, swept
ib_minutes in [15,30,45,60,90] x target_r in [0.5-2.0] on 2021-2024, see
src/tune_ib.py.

Result: best in-sample config (ib_minutes=30, target_r=2.0, 1008 trades --
2.35x the baseline's trade count) put up net $177,070 but Sharpe only 1.20,
LOWER than the plain VIX-band baseline's 1.44 even in-sample. Higher raw
profit was mostly just from trading far more often, not from being better
per trade. Holdout (2025-2026, unmodified) was a clear failure: net
**-$14,641** (losing money), max_dd **-$102,821** (worse than every other
strategy tested in this project, over 2x the baseline's), Sharpe -0.15.

Why: the IB range (first 30-60min) is much narrower than the full VIX band,
so it triggers far more often -- most IB breakouts are likely false
breakouts/noise without additional confirmation. Decoupling a tight
IB-based stop from a wide VIX-sized target also creates a structurally
asymmetric small-risk/big-reward trade population that's highly exposed to
repeated whipsaw stop-outs, which is exactly what 2025-2026 punished. Simply
using a different (non-VIX) entry trigger wasn't enough on its own --
trigger FREQUENCY/QUALITY matters as much as trigger SOURCE, and this one
traded too often on too little confirmation.

### 6. Trend-direction filter using NQ's own price trend (not VIX)
Shorts dominated 2022 (downtrend), longs helped in 2025 -- a real pattern
never actually explained, since every filter tried conditioned on VIX level/
em_points, which the meta-finding shows can't separate trend from whipsaw.
Bias/filter trade direction using NQ's own trend (e.g. price vs 50-day
average) instead -- a genuinely different information source.

### 7. Opening-range compression as a trade-quality filter
~80% of breakout trades are near-zero-P&L EOD chop; all the edge sits in the
20% that hit target/stop. The divergence-signal idea tried to predict which
using a VIX-derived ratio and failed on holdout. Try a price-action filter
instead: narrow opening range (vs its own recent history) precedes real
breakouts; wide/choppy opens precede whipsaw. Different information source
than anything tried against this same problem so far.

### 8. Weekly (not daily) systematic premium selling
The iron condor's failure doesn't kill the VRP premise, just this
operationalization (entry at the band strike, 1-day hold, unconfirmed).
Standard practice sells weekly/monthly, fixed-delta strikes (not tied to one
day's band), more time for theta, tail risk diversified across fewer/larger
positions instead of many 1-day bets. Directly answers whether VRP-selling
itself is wrong, or just our specific version of it.

## Idea 9 (user-supplied): MNQ ORB + VIX target + runner -- TESTED, WEAK POSITIVE (holdout survived, degraded)
User supplied a reference script (5-year MNQ opening-range breakout: 09:30-
09:55 ET range, close above range high triggers a long at the next bar's
open, stop at range low, fixed 1R target, several position-sizing schemes
including Harvey target-vol and Moreira-Muir inverse-variance). Combined
with this project's core idea: target sized off VIX-implied em_points
instead of fixed 1R, plus a "runner" -- partial exit at the VIX target, the
remaining half trails (highest high since entry - trail_r * em_points).
Long-only, matching the reference script. Implemented in
src/orb_vix_runner.py (MNQ.V.0 1-min bars resampled to 5-min, reusing this
project's VIX cache), swept in src/tune_orb.py (target_r x trail_r, runner
on) and src/tune_orb_norunner.py (same sweep, runner off, as a stress test).

2021-2024 in-sample sweep found target_r=0.75, trail_r=0.25 as a genuine,
smooth local peak (Sharpe 0.99, confirmed stable under a finer 0.55-1.0 grid
-- not a coarse-grid fluke). At the naive default target_r=1.0, the VIX
target was reachable on only 19/610 trades (3%) -- the runner mechanic was
essentially inert and the strategy was behaving as plain ORB-with-time-exit
in disguise; a control run with VIX-target and runner both OFF (plain fixed
1R off the ORB range, matching the reference script exactly) scored Sharpe
0.76, confirming this. The no-runner stress test confirmed target_r=0.75
beats target_r=1.0 (Sharpe 0.95 vs 0.75) independent of the runner, so the
runner isn't what drives that result.

HOLDOUT CHECK (2025-2026, target_r=0.75/trail_r=0.25 unmodified from
in-sample): Sharpe fell from 0.99 to **0.62** (real degradation, ~37%
relative), but win rate held (56.0%, even slightly above in-sample),
profit factor stayed >1 (1.15), and net P&L was solidly positive ($16,683
over 268 trades) with drawdown similar in magnitude to in-sample. This is
the FIRST idea in this project (of #1, #3, #5, and this one) whose
in-sample-optimized config did NOT fail or go negative on holdout -- every
other refinement attempt collapsed completely. Still meaningfully weaker
than the original validated NQ VIX-band breakout baseline's holdout Sharpe
of 1.06 on the same 2025-2026 window, so this isn't an upgrade over the
existing strategy -- but it's a second, independently reasonable strategy
(different instrument: MNQ; different mechanism: ORB entry rather than
VIX-band breakout) that shows real, if weaker, signal. Worth treating as a
genuine secondary candidate rather than closing out negative like the
others.

## Idea 10 (user-supplied): Overnight-range-midpoint directional filter -- TESTED, FAILED ON HOLDOUT (inverted)
User supplied a stat: mark the overnight (18:00-09:30 ET) range's high/low/
midpoint; if RTH opens above the midpoint the overnight high tends to break
first (76.2% historically), if below the overnight low tends to break first
(75.6%). Applied as a pre-trade filter on the long-only ORB strategy: only
take the ORB long breakout when RTH opened above the prior overnight
midpoint (the side the stat says the flow already leans toward). Implemented
in src/orb_overnight_filter.py (overnight window + midpoint-side
classification) and src/tune_orb_filtered.py / src/check_combined_holdout.py
(combined with VIX-target+runner).

2021-2024 IN-SAMPLE: the cleanest-looking result in this entire project.
Win rate nearly identical across "above" (55.3%) and "below" (55.2%)
groups, but Sharpe/profit-factor diverged hugely: above=0.85 PF=1.25,
below=0.23 PF=1.08, unfiltered=0.76. Identical win rate with diverging
risk-adjusted return is the textbook signature of a real quality filter
(predicting trade SIZE/quality, not win/loss), not noise -- looked far more
credible than anything else tested. Combined with VIX-target+runner
(target_r=0.75/trail_r=0.25): did NOT compound -- above-filtered combined
Sharpe was 0.92, WORSE than unfiltered VIX+runner's 0.99, because the
VIX-target+runner mechanism already rescues a lot of the "below" trades'
profitability (their contribution roughly doubled, $6,594 -> $14,820, once
under smarter exit management) -- the filter and the smarter exits turned
out to be partially redundant, not additive.

HOLDOUT (2025-2026, unmodified thresholds/logic): completely failed, and
inverted. Plain ORB: above=0.62 Sharpe, below=**0.86** Sharpe (below beat
above -- the opposite of the in-sample and stat's predicted direction).
Combined with VIX-target+runner: above=0.46, below=0.43 (no real
separation either way, both below the unfiltered 0.62 baseline). Despite
being the most statistically convincing in-sample result in this project
(same win rate, diverging risk-adjusted return -- not just "found a lucky
subgroup"), it still completely failed to generalize, even flipping
direction. Strong reminder that a mechanistically sensible, clean-looking
in-sample split is still not sufficient evidence with only ~600 in-sample
trades. Do not use this filter. Best validated ORB config remains
VIX-target+runner unfiltered (target_r=0.75, trail_r=0.25): Sharpe 0.99
in-sample, 0.62 holdout (idea #9).
