# Model and decision-logic audit — November 15, 2026

## Assessment

The current bot should not be treated as a validated player-prop probability
model. Its calibration currently discards the magnitude of the player-model
signal, the ensemble promotion guard does not execute, and selection can admit
negative-EV recommendations. Several implemented contextual adjustments never
reach the probabilities used to choose bets.

This audit follows the actual pregame path from BDL logs through minutes,
Bayesian rates, ensemble predictions, distributions, calibration, ranking and
alerts. Training, calibration promotion, historical backtesting and selected
live-model paths were also reviewed. Database analysis used read-only
PostgreSQL connections. No betting thresholds, prediction logic, fitted models,
production records or running services were changed in this follow-up audit.

Severity below reflects correctness and the risk of drawing false conclusions,
not a claim that any one defect explains the entire recorded loss.

## 1. Critical: a negative expected-value recommendation can clear selection

Sources: `src/models/edge_ranker.py:161`,
`src/pipelines/scan_props.py:2353`, `src/pipelines/send_alerts.py:143`.

Ranking calculates two different quantities:

- Selection edge: `model_prob - devigged_implied_prob`.
- Return at the offered price: `model_prob * decimal_odds - 1`.

The filter tests the first but does not require the second to be positive.
Sorting by break-even edge does not reject a losing proposition. Devigging the
best over and under from potentially different books also does not produce a
single book's coherent fair-price estimate.

Offline reproduction: probability 53%, odds 1.80, devigged implied 45% produces
an 8% selection edge, clearing a 4.33% floor, despite **-4.6% expected return**.
The sender assigns a zero stake but continues persisting and delivering the
recommendation.

Production confirmation: **75 alerts from October 22 through November 14 have
`model_prob * odds <= 1`; all 75 have $0 stakes.** Sixty occurred from November 9
onward. These were not recorded cash losses, but they pollute the recommendation
history, win rates and calibration population. There are 79 zero-stake alerts
in total over the same period; zero stakes alone are not proof of negative EV.

Repair: require valid finite probabilities/prices, strictly positive EV and a
positive allowable stake at the final send/persist boundary. Define thresholds
against the actual offered price; retain fair-market disagreement as a separate
diagnostic.

## 2. Critical: the ensemble promotion guard is always skipped

Source: `src/models/ml_model.py:604–663`.

`EnsembleModel.train()` calls `_gl(...)` at line 611, but `_gl` is imported only
inside the outer exception handler at line 663. Python treats it as a local
variable throughout the function. This raises `UnboundLocalError` before the
baseline is read. The inner `except Exception: pass` swallows it and training
continues to save the candidate.

The offline reproduction substitutes inexpensive learners and intercepts disk
writes while executing the real training control flow: training returns True,
the baseline database constructor is called **zero times**, and model save is
called once.

Even after fixing the logger, that guard uses the SQLite constructor directly
instead of the active database factory, and compares different runs' RMSEs
rather than candidate and incumbent on the same untouched evaluation period.

Repair: initialize logging normally, make promotion failures explicit, and
evaluate both artifacts on the same frozen, time-ordered dataset before any
active artifact replacement.

## 3. Critical: training validation does not establish future betting performance

Sources: `src/models/ml_model.py:354`, `:460`, `:584`, `:604`, `:788`;
`src/pipelines/train_ml.py:119`.

- The ensemble uses shuffled KFold. Future games and overlapping player-history
  windows can appear in training for earlier validation targets.
- The Ridge meta-learner is fitted on OOF predictions, then its score is
  calculated on those same meta-training rows. Base-learner OOF predictions do
  not make this an independent test of the completed ensemble.
- The standalone XGBoost split takes the last 20% of a matrix assembled by
  appending each player's history. It is a player-order split, not a global
  chronological holdout.
- Historical games across multiple seasons receive today's team/opponent
  context. That context was not available before those historical games and
  does not describe the old opponent roster or defense.
- Selecting only currently active players adds survivorship bias.
- Training targets stat production per actual minute. Low-minute appearances
  can have noisy rate labels, and rate RMSE is not a test of probability
  calibration, available prices, or betting EV.

Repair: preserve event IDs and target timestamps in every training row; use
historical as-of context and forward splits grouped by game/date; train stacking
only on predictions made from strictly earlier training periods; reserve a
separate final evaluation period. Compare projection error, probability scores
and decisions at recorded prices. Do not regard the current guard as evidence
that retraining improves the strategy.

## 4. High: final betting probabilities currently ignore model confidence

Sources: `src/models/calibration_market.json`,
`src/models/calibration_model.py:91`, `:205`.

The current fitted file has `k = 0.0`, `b_under = 0.1577`, `n = 773`:

`logit(p) = logit(1 / odds) + k * (logit(raw) - logit(1 / odds)) + b_under * UNDER`

At a 50% book-implied probability, raw under probabilities of **60% and 90% both
become 53.934%**. Overs become book-implied probability. The raw model still
controls the `MIN_RAW_EDGE` eligibility check, so it has not disappeared from
selection altogether. Past the gate, however, changing matchup, usage or
player confidence cannot change probability when price and side stay fixed.

This can be a reasonable conservative calibration result; it is also direct
evidence that the fitted calibration did not support positive weight on model
disagreement in its training sample. It should not be described as a strong
player-specific edge.

Repair: treat the price-plus-side rule as a separately named baseline. Validate
it and the raw player model independently on later games before changing `k` or
adding more contextual boosts. Manually increasing model weight would bypass
the evidence rather than repair it.

## 5. High: calibration promotion and training population are weak

Source: `src/pipelines/train_calibration.py:177–239`.

The trainer computes market Brier but does not require the candidate to beat
it; promotion only compares with the incumbent. On nightly reruns, the
incumbent can already have been trained on part of the next run's held-out
slice. Reusing a moving test slice is not prospective validation. Sorting by
settlement timestamp also allows late-settled old games to land in a later
slice.

The fit uses only alerts, including zero-stake entries, while live eligibility
is additionally conditioned on raw edge, time, market floors and sharp vetoes.
With overs disabled, the new calibration observations are also predominantly
unders. This limits what the fit can establish about the whole prop universe
or re-enabling overs.

Repair: freeze evaluation periods by event date, record artifact versions and
fit cutoffs, score candidates and incumbent on an unconsumed future period, and
require a baseline comparison. Separate executable picks from diagnostic
observations and paper picks.

## 6. High: variance controls do not affect the main simulation

Source: `src/models/distributions.py:417–485`.

`get_probability_distribution()` accepts `variance_scale` but does not pass it
to `monte_carlo_over_under()`, whose signature has no such parameter. Thus
playoff/starter/bench variance adjustments in `scan_props.py` do not affect the
MC path. The active path commonly uses MC because missing BDL foul data falls
back to a positive foul rate.

With a fixed RNG, changing variance scale from **1 to 4** leaves both outputs
identical: over **0.2204**, under **0.7796**.

The MC dispersions are hard-coded market constants, rather than estimated
conditional residual distributions. Every alternate line gets a new unseeded
simulation, adding decision noise and making replay difficult. A 5,000-draw
estimate near 50% has about 0.7 percentage points of Monte Carlo standard error;
that is numerical uncertainty, in addition to model uncertainty.

Repair: use a shared distribution/sample set per player-market-context, expose
and test dispersion inputs, record RNG/version information, and estimate
dispersion from genuinely held-out residuals.

## 7. High: fallback probabilities can ignore the adjusted projection

Sources: `src/models/distributions.py:89`, `:487`.

For points and PRA, the fallback bootstraps raw historical box scores and never
recenters them to the supplied projected mean. A role/minutes/injury adjustment
can change the reported projection while barely affecting betting probability.

Offline example with historical outcomes alternating 10 and 20 and line 15.5:
projected means **5 and 30 both return 50% over / 50% under**. This is a
fallback-path defect; the active foul-context path generally uses MC instead.

The bootstrap also adds random resampling noise to an empirical exceedance
fraction that could be computed directly.

Repair: choose an explicit residual or rate distribution centered on the
adjusted projection, or label the empirical historical hit rate separately.

## 8. High: the intended BDL rotation model fails before producing minutes

Sources: `src/clients/bdl_game_logs.py:152–218`,
`src/clients/rotation_model.py:91`, `src/pipelines/scan_props.py:1796`.

BDL logs keep game IDs in a separate cache but omit the `GAME_ID` dataframe
column. The rotation model indexes `logs['GAME_ID']`. In the BDL path this
raises KeyError, and the scanner silently catches it, leaving rotation minutes
at zero and using the heuristic fallback.

BDL logs also omit `PF` and `SEASON_ID`. Consequences:

- Training supplies `player_foul_rate = 0`; live inference defaults it to
  **0.083** for the same missing data. This is a concrete train/live mismatch.
- Foul simulation uses a generic rate rather than that player's actual rate.
- Playoff-share features and playoff/series helpers cannot work as intended
  from those logs.

Repair: define one canonical log schema for fresh and cached rows, preserving
game identity, personal fouls and game type. Assert feature parity across
training and inference. Make fallback reasons observable.

## 9. High, latent: rotation allocation is not a feasible lineup model

Source: `src/clients/rotation_model.py:116–164`.

The next-man rule picks the highest slot probability among every non-absent
player, including starters already occupying other positions. It does not
model five simultaneous court slots or positional eligibility. Multiple absent
players can independently award the same player overlapping minutes. Ties
can award the same vacancy to multiple players.

A valid input with a 24-minute target and two absent players returns **72
projected minutes**. This is currently partly masked by the BDL `GAME_ID`
failure. Current stored candidate minutes did not exceed 48; the reproduction
demonstrates a latent constraint failure, not an observed 72-minute alert.

Repair: enforce player/slot capacity and five players per slot, allocate lost
minutes jointly, and check team regulation minutes sum to 240 before applying
overtime assumptions. Fixing the missing column alone would expose this defect.

## 10. Medium: injury probability, minutes and conditional play are conflated

Sources: `src/models/projections.py:218–293`, `:349`;
`src/models/edge_ranker.py:184–207`.

The heuristic multiplies questionable minutes by 0.75 and doubtful minutes by
0.35, then the ranker applies another injury penalty. A player who does not
play is normally voided by settlement. Therefore availability probability
should not automatically be converted into reduced production conditional on
playing. A genuine minutes restriction is a separate state.

Using a positive rotation override also bypasses the heuristic's injury
scaling entirely, so fixing rotation changes the injury behavior as a side
effect. The OUT filter remains in the ranker.

Repair: represent availability/void, conditional minutes restriction and
in-game exit risk separately. Estimate conditional outcome probabilities for
the wager's actual settlement rules.

## 11. Medium: some contextual adjustments are inconsistent or miswired

Sources: `src/models/distributions.py:249–356`,
`src/pipelines/scan_props.py:1752–1792`, `:1952–2118`.

- Standings minutes adjustment changes `projected_minutes` after the mean and
  ML blend are calculated. MC recenters its rate back to the unchanged mean,
  so a supposed minutes reduction does not consistently reduce expected stats.
- The MC `next_opp_win_pct` input is populated with **tonight's opponent's**
  standings percentage, not the next scheduled opponent. The look-ahead rest
  narrative therefore drives the wrong input.
- Blowout, rest, fatigue and foul adjustments appear at several layers. Some
  overlap historical averages or the ML feature effect; others are canceled
  by recentering. There is no attribution or ablation evidence for the net
  effect of the combined adjustments.
- With two initiators absent, the largest observed `rate_without` can be
  multiplied by 1.50, an unvalidated extrapolation from a single-absence split.
- The regular-season 50/50 Bayesian/ML blend is fixed rather than selected on
  held-out performance. Its contextual inputs also differ from historical
  training context.

Repair: give minutes and rate one explicit owner each, build an auditable
factor breakdown, and enable adjustments only after evaluating their incremental
effect. Do not assume a plausible basketball story establishes an edge.

## 12. Medium, latent: integer-line pushes are counted as under wins

Sources: `src/models/distributions.py:23–40`, `:67–84`, `:109`, `:409`.

The discrete functions implement under as `P(X <= line)`, or simply
`1 - P(X > line)`. At integer lines the equality is a push, not an under win.
For Poisson mean 1 and line 1, the code returns under **73.58%**; the actual
under-win probability is **36.79%**, push is **36.79%**, and under conditional
on no push is **58.20%**.

There were no integer lines in the inspected current-season alert/candidate
sample, so this does not explain its losses. It matters before adding new feeds
or whole-number alt lines.

Repair: return win/loss/push probabilities explicitly, or consistently condition
on no push for pricing while retaining the non-push frequency for expected return.

## 13. Medium: probability caps break a coherent distribution

Source: `src/models/distributions.py:446–459`.

Over and under are capped independently. On half-point lines they should sum to
one, but an offline case produces over **0.0000201**, under **0.93**. The missing
mass has no meaning. For zero projected mean the wrapper returns under 0.75
rather than a valid degenerate distribution or an explicit invalid-projection
status. Separate calibration of sides adds further coherence concerns.

Repair: distinguish invalid inputs from valid distributions; apply any
confidence shrinkage jointly to complementary probabilities and test
normalization, bounds and monotonicity across lines.

## 14. Medium: adaptive thresholds and timing use weak evidence

Sources: `src/models/edge_ranker.py:33–125`,
`src/data/db_postgres.py:406`, `src/clients/bdl_bridge.py:38`,
`src/pipelines/scan_props.py:920`, `:2302–2367`.

Adaptive floors use **vigged** CLV despite the availability of no-vig CLV.
The earlier audit found positive vigged CLV alongside negative no-vig CLV, so
the adaptation can reward the wrong signal. Floors also relax near tip by a
fixed schedule, without demonstrated time-bucket performance. General
`EDGE_MIN` is not a guaranteed floor when a candidate already has the dynamic
floor field.

The BDL spread/total cache lasts 24 hours, and sharp prices are intentionally
cached for budget conservation. The probability/ranking objects do not enforce
a unified freshness requirement on all inputs. The pre-tip check happens before
the potentially lengthy game scan, while the final alert filter uses previously
computed hours-to-tip rather than checking the clock again. Missing or malformed
start times fail open.

Repair: record source timestamps with each input, require a final executable
price/start-time check, fail closed on unknown timing, and evaluate thresholds
using actual-price EV and no-vig CLV. Balance freshness against API cost
explicitly.

## 15. High for evaluation: historical backtest does not replay this strategy

Source: `src/pipelines/backtest.py:1–151`.

The backtester invents the line from a rolling historical mean and assumes 50%
implied probability. It does not replay actual historical prices, the full live
ML/context path, current calibration, raw-edge eligibility, sharp vetoes,
injury/news information, timing gates or execution constraints. Its default
projection context also differs from live context. It is a synthetic projection
experiment, not evidence of betting profitability for the current bot.

The candidate log is more useful because it records actual prices and
predictions, but it refreshes a row to the latest scan. It cannot reconstruct
every prior alert decision or news-triggered update. Artifact version, input
freshness and reason codes are also needed for exact replay.

Repair: preserve immutable decision snapshots and replay the same selection and
sizing functions against prices available at each historical decision time.

## Recorded outcomes: where the model stands

The broader settled candidate sample covers November 9–14: **8,030 candidate
sides**. These contain repeated players, games, alternate lines and opposite
sides; sample counts are not independent observations.

| Market | Candidate sides | Raw Brier | Calibrated Brier | Offered-price baseline Brier |
| --- | ---: | ---: | ---: | ---: |
| Points | 2,186 | 0.26993 | 0.25067 | 0.25004 |
| Rebounds | 1,476 | 0.25247 | 0.24595 | 0.24614 |
| Assists | 1,118 | 0.25139 | 0.24404 | 0.24383 |
| Threes | 1,051 | 0.25889 | 0.24609 | 0.24453 |
| Blocks | 168 | 0.24167 | 0.23286 | 0.23213 |
| Steals | 416 | 0.22955 | 0.22679 | 0.22638 |
| PRA | 1,615 | 0.26592 | 0.25033 | 0.25014 |

Lower Brier is better. Offered-price baseline is `1 / odds`, including vig;
it is not a claim that those prices are true probabilities. The raw model
trails this baseline in every listed market. Calibrated rebounds are marginally
better descriptively, without establishing a statistically reliable edge.

Among candidate sides with raw probability at least five points above
break-even:

| Side | Rows | Average raw probability | Average calibrated probability | Win rate | Flat-unit ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Under | 1,733 | 68.21% | 56.89% | 52.45% | -1.91% |
| Over | 393 | 59.81% | 50.83% | 49.36% | -1.20% |

These are observational replays of stored candidate snapshots, not a new
independent validation of today's calibration parameters. They nevertheless
show why high raw confidence should not be trusted as a large edge.

Using one latest settled observation per event/player/market to reduce
line/side repetition, average projection-minus-actual bias is approximately
**-0.52 points**, **-0.14 threes**, **+0.19 steals**, and **0.00 rebounds**.
An accurate average mean can still produce inaccurate tail probabilities;
recalibrating only means will not resolve the distribution and decision issues.

## Repair order

1. Fix final selection validity: positive actual-price EV/stake, current start
   time, current price, and explicit rejection reasons. Keep diagnostic rows
   separate from executable recommendations.
2. Repair the canonical BDL schema and train/live feature parity. Resolve
   rotation feasibility before activating its override.
3. Make probability generation coherent: push handling, projection-centered
   fallback, connected variance controls, shared simulations and invariants.
4. Rebuild training validation and artifact promotion with as-of features,
   game/date splits, a frozen baseline and explicit failures. Fixing the logger
   alone is not enough to make the current guard valid.
5. Record immutable prediction/decision snapshots, then evaluate a simple
   baseline, Bayesian model, ML blend and each contextual adjustment separately
   on later games. Only then choose calibration, blend weights and market floors.

The evidence supports these correctness repairs before strategy expansion.
Changing market thresholds or increasing model confidence to generate more
plays would not address the underlying problems.

## Reproduction

From the `nba-prop-bot` directory:

```powershell
..\.venv\Scripts\python.exe -m scripts.audit_model_logic
```

The diagnostic makes no API calls. It fixes random draws for comparisons,
uses synthetic inputs, substitutes inexpensive learners for the training
control-flow probe, and intercepts all model saves. It reproduces current
behavior without changing fitted artifacts. It is a diagnostic audit script,
not a claim that the defects have been fixed or a replacement for future
regression tests.
