# Projection and probability correctness patch

## Data and feature consistency

Fresh and cached BDL logs now preserve GAME_ID, SEASON_ID and PF, with provider
identity recorded on the dataframe. Game type is derived only when the feed
supplies a postseason flag; missing type is explicitly unknown. Missing fouls
remain missing rather than being fabricated as zero. Non-final game rows are
excluded when status is supplied. Recent game IDs follow sorted game dates,
not API response order.

SQLite migrations 27/28 and PostgreSQL initialization add nullable pf and
season_id cache columns. Legacy cache entries without the new schema are
refetched, including permanent historical caches. No production cache was
rewritten during implementation.

Training and live feature construction use the same personal-foul calculation
and historical window. Unknown fouls use the same explicit 0.083 fallback;
observed zero fouls remain zero. NBA on/off and Redis lookups resolve NBA player
IDs even when box scores come from BDL.

The corrected feature semantics use schema version 2. Existing unversioned
ML artifacts are not compatible and inference falls back to the Bayesian
projection. The files remain untouched. Newly saved artifacts carry the
version, but **version compatibility does not establish model quality**:
the training/validation and promotion defects identified in the audit remain
step 3 work. No retraining was run.

## Minutes and rotation

Rotation allocation now handles all absences jointly, with at most one court
place per player per two-minute slot and five places per slot. Tied eligible
replacements share vacancies; players already at full slot occupancy cannot
absorb more. Each player is bounded by 48 regulation minutes and aggregate
allocation by 240. Insufficient rotation depth leaves minutes unallocated
rather than inventing additional players or overfilling slots. This is a
constrained marginal allocation, not a validated lineup/position model.

BDL player/game IDs are not NBA play-by-play IDs. That path now falls back
explicitly instead of indexing a missing column or sending the wrong IDs to
the NBA endpoint. BDL rotation overrides remain unavailable until mapping
and source validation are implemented. The pre-existing raw NBA PBP parsing
and lineup reconstruction still need a separate review before relying on
rotation quality; allocation bounds do not validate source templates.

Heuristic/override minutes and the final scanner minutes are also capped at
48. Final capping rescales expected production consistently. Standings-based
minutes changes now adjust the projected mean as well as minutes.

## Probability contract

`prob_over` and `prob_under` are conditional on no push and sum to one.
`prob_push` separately describes the probability of a returned stake.
Unconditional over win probability is
`(1 - prob_push) * prob_over`, and similarly for under. These three
unconditional outcome probabilities sum to one.

Poisson, negative binomial, zero-inflated Poisson, rounded Normal, empirical
fallback and Monte Carlo now distinguish equality from an under win. The
scanner carries push probability into candidates and the selection guard
rejects a certain push. Conditional pricing preserves the existing break-even
test; expected return per original stake for an integer line is
`(1 - prob_push) * (prob_over * odds - 1)` for an over. Its sign is the same
as the conditional return used for eligibility. Existing displayed EV remains
the conditional value.

Independent side caps were replaced with complementary clipping. A valid
zero-mean distribution is handled explicitly. Nonfinite/negative main inputs
and invalid variance scales are rejected.

## Distribution behavior

The points/PRA empirical fallback is centered on the adjusted projection.
Historical residuals are scaled by variance_scale; a location adjustment
preserves the target mean even after enforcing nonnegative production.
Analytic randomized rounding gives integer-valued support without adding
random resampling noise. Tests recover the target expectation from the full
tail distribution, including a near-zero mean case.

Monte Carlo now receives variance_scale and applies it to excess-Poisson
dispersion. This scales dispersion, not every source of total variance: foul
and blowout minute variation and the Poisson variance floor remain distinct.
A fixed default seed produces the same scenarios for repeated calls and
alternate lines with identical context. This removes per-line resampling
inconsistency; it does not eliminate finite-simulation error or validate the
hard-coded dispersion assumptions.

## Verification and activation

Regression tests cover fresh/cache parity, legacy cache invalidation, provider
identity, date ordering, foul feature parity, artifact compatibility, minute
capacity/ties/absences, integer pushes, zero means, probability normalization,
projection sensitivity, expectation preservation, connected variance settings,
and deterministic monotonic alternate lines. Existing selection tests also run
against the updated probability contract.
Final full-suite result: **254 passed, 2 pre-existing failures** in institutional
live-model tests (portfolio rounding and enriched-versus-basic win probability).

No live API requests, Telegram sends, training jobs or production schema
migrations were executed for this patch. The running scheduler has not been
restarted. Normal initialization applies cache migrations on restart.

These are correctness changes, not evidence of improved betting performance.
Existing calibration and strategy floors were preserved; future validation
and model promotion are the next phase.
