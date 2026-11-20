# Forward validation — activated November 15, 2026

The updated bot is running under `run_bot.ps1` supervision. Following the final
delivery fix, the supervisor started at approximately 9:34 AM Eastern and the
scheduler reported live at 9:35 AM. Seven candidate experiments are enrolled:
points, rebounds, assists, threes, blocks, steals, and market-anchored calibration.
All seven active incumbent files still match their pre-training fingerprints.
No candidate was promoted.

## Initial candidates

Six ensembles were actually fitted and staged from the existing BDL cache:
66,640 historical rows, with 2,829 dated current-season targets per market.
Earlier games remain feature history, while targets start October 1, 2026.
The final historical test period ends November 13 and was not used for fitting.
The cache's missing foul/game-type fields are disclosed in each manifest;
these initial candidates are hypotheses, not validated betting improvements.

Calibration was separately staged from eligible historical settlements.
Seventy-four rows whose event IDs were assigned to multiple dates were
excluded from training and counted explicitly in its manifest. The underlying
alerts and results were not deleted or changed by that exclusion.

## Recording

The scanner records the first verified pregame quote and paired incumbent/
candidate prediction for each experiment, event, player, market, line and side.
Later scans cannot replace the original book, price or probabilities. Both
over and under observations can be recorded without generating recommendations,
stakes or executions. Existing recommendation gates remain separate.

Snapshots include raw and calibrated probabilities, means, exact timestamps,
book prices, game/player identities, candidate/incumbent hashes and the inputs
used to construct the prediction. A separate immutable context manifest
fingerprints serving source files, active artifacts and relevant settings.
Database triggers reject updates or deletes to predictions, contexts and results.
Inserts also require an unexpired quote/game deadline at database execution time.

Candidate rates are evaluated through the configured blend, downstream scalar
adjustments, distributions and current calibration for their own market.
Calibration candidates are evaluated on the incumbent raw probabilities.
This is per-market comparison; it does not independently validate every
cross-market effect, correlated bet portfolio or arbitrary combination of
separately promoted models. PRA is observed for the global calibration
candidate, while the initial rate candidates are the six component markets.

The active experiment remains frozen when a newer candidate is staged.
Changing serving code, settings or active artifacts pauses an old context and
starts a separate experiment. After a promotion, remaining experiments must
collect under the changed serving context rather than mixing old snapshots
with the new baseline.

## Settlement and evaluation

The existing settlement job uses its batched final box scores to append
prospective results separately from predictions. Matching requires the correct
final game and player ID, or an exact matchup/name when IDs are unavailable.
Missing/ambiguous data remain pending. An explicit zero-minute stat row is
voided. Results are not graded before tipoff, and invalid/missing required
statistics are not fabricated as zero production.

Pending prospective dates remain in the settlement backlog beyond the legacy
four-day alert retry window. Evaluation uses complete event-date prefixes;
it cannot pick only the already-settled winners from an incomplete day.

The scheduler evaluates cohorts at **5:15 AM Eastern**, after the 4 AM
settlement job. The gate requires at least thirty non-push observations across
ten games, game-weighted Brier improvement over both the incumbent and offered
price baseline, and no log-loss deterioration. Used evaluation dates cannot
be recycled for another candidate targeting that active artifact. Diagnostic
raw Brier scores remain available when the active k=0 calibrator suppresses
differences in final probabilities.

`MODEL_SHADOW_ENABLED=true` and `MODEL_SHADOW_AUTO_PROMOTE=true` are active by
default. Automatic replacement is possible only after the gate passes; no
historical holdout, missing price or insufficient cohort can authorize it.
The sample minimum is an initial checkpoint, not proof of a durable edge or
guaranteed profit. After a rejected cohort, retraining/new candidate staging
remains a separate task; automatic rate retraining is still disabled.

## Delivery and operation

Activation found 617 unsent prop messages from November 14 in the legacy
digest queue. The digest now ignores prop recommendations entirely because
their text has no verifiable current quote or expiry. Those records remain
preserved and are not marked as successfully delivered. New prop delivery
continues through the expiring instant path implemented in step 1.

The current configuration scans every 120 minutes during its existing scan
window starting at 11 AM Eastern. Low Odds API credit availability does not
disable the scheduled BDL scan path; it can limit sharp-reference coverage.
BDL briefly rate-limited startup injury sync, which retried successfully.
Redis and the optional live websocket are not required by this validation path.

The PostgreSQL schema loader was corrected to preserve dollar-quoted function
bodies when splitting SQL statements, allowing immutability triggers to install.
Their presence was checked in the live database, and a rolled-back probe update
was correctly rejected. Snapshot insert/deadline syntax was also checked using
read-only EXPLAIN. No fake prediction or historical outcome was inserted into
the production prospective record.

## Inspection commands

Run from `nba-prop-bot` using its production virtual environment:

```powershell
venv\Scripts\python.exe main.py shadow_status
venv\Scripts\python.exe main.py shadow_evaluate
```

`shadow_status` shows collecting, paused and completed experiments and their
prediction/settlement counts. `shadow_evaluate` uses only the recorded database
snapshots and can promote if all configured requirements pass. The manual
`promote` evidence-file command remains available; automatic evaluation does
not rely on arbitrary exported JSON timestamps.

## Verified state

At the final pregame check there were seven active experiments, zero recorded
prospective predictions and zero settled prospective outcomes. All seven gate
results were `insufficient_prospective_prices`. This is expected before the
scan window; meaningful validation needs future games, not retrospective data
relabeled as prospective observations.

Final tests: **294 passed; two pre-existing institutional live-model tests
still fail** (portfolio rounding and enriched-versus-basic win probability).
New tests cover immutable first-price recording, context changes, candidate
freezing, freshness/tipoff checks, strict settlement identity, incomplete-day
exclusion, blend calculations, SQL-body parsing and suppression of legacy
prop digests. The scheduler, supervisor and child process were verified live.
