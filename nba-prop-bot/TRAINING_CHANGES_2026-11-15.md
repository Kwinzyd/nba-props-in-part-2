# Training validation and promotion changes

## Training is separate from deployment

Both standalone XGBoost and ensemble training now write immutable candidate
directories beneath `src/models/.../candidates/`. They do not overwrite active
artifacts or replace the model object currently used for inference. Each
candidate has a manifest with artifact SHA-256, incumbent fingerprint,
creation time, fit/test cutoffs, sample counts and diagnostic scores. Rate
models also record a fingerprint of their input matrix, labels and metadata.

The former ensemble guard, its uninitialized logger, silent exception path,
SQLite baseline lookup and comparison with unrelated prior-run RMSE are gone.
The standalone guard that could continue after evaluation failure is gone too.
Missing timestamps, invalid features, insufficient dates or fit failures now
produce an explicit failed stage and preserve the incumbent.

`train` success means a candidate was staged, not promoted. Scheduled anchored
and legacy isotonic calibration routines also stage candidates; they return
False to indicate that active calibration was not replaced. Their prospective
evaluation and promotion use the same gate as rate-model candidates.

## Chronological validation

Every training row carries event identity and target game date. Logs are
explicitly sorted by timestamp, and feature history excludes the target and
all same-date games. The shared splitter groups all players/markets for a game
and every game on a date together, even when input rows are in player order.

Partitions use approximately 60% / 20% / 20% of unique dates. Ensemble bases
fit only the first partition; the Ridge combiner fits predictions on the
second, strictly later partition. The third partition is untouched until
evaluation. Standalone XGBoost and calibration fit on the first two partitions
and evaluate on the third. There is **no refit on the test outcomes** after
scoring. The staged artifact is the evaluated artifact.

Minimum checks require ten distinct dates and, for rate training, at least
50 base-training rows, ten middle-partition rows and twenty final-test rows.
Historical rate MSE is compared with a prior-history ten-game rate baseline.
These scores describe projections; they do not certify a betting edge.

## Features available before the game

Feature schema version 3 excludes current-season opponent tables and live
tracking/playtype profiles without historical as-of archives. Historical
training no longer fetches today's team context for old games. The retained
usage and true-shooting features always use their historical box-score
calculations, including at inference, rather than switching to a different
live profile definition. Current contextual adjustments in the separate
Bayesian projection are outside this ML feature-schema change.

Outcome-derived CLV sample weighting is disabled pending evidence of an
incremental benefit. The legacy CLV training entrypoint stages ordinary
chronological candidates. Training still selects currently active players;
this population limitation and survivorship bias have not been eliminated.

Previous feature-schema artifacts remain on disk but are incompatible with
the new feature set. Baseline inference remains available. No fitted artifact
was retrained or promoted during this implementation.

## Prospective price-aware promotion gate

`promote_candidate()` accepts a manifest and recorded, paired predictions with
settled outcomes. It requires:

- Candidate and incumbent identities matching the frozen manifest and current
  files; unchanged candidate bytes.
- Evaluation dates after fitting, the historical diagnostic test period, and
  any previously consumed evaluation dates for that active artifact.
- Predictions made after candidate creation and before tip; settlement after
  tip and no later than the current time. NBA event dates use Eastern time.
- Valid odds, probabilities and binary non-push outcomes; no duplicate
  event/player/market/line/side rows or inconsistent event dates.
- At least thirty paired observations across ten games by default.
- Candidate Brier score strictly better than **both** the incumbent and
  `1 / offered odds`, with log loss no worse than either. Metrics weight
  games equally so correlated sides/lines do not become independent games.

If there is no incumbent file, the price baseline is used for both comparisons.
Price probabilities include vig: this is an available-price baseline, not an
assertion that the book offers true fair probabilities. Passing these finite
sample checks is not a statistical guarantee or a profitability claim.

Completed evaluation cohorts are consumed on rejection too. Adjusting a
candidate and retrying the same dates cannot make that test set fresh again.
A later new cohort may be evaluated. A per-active-path process lock prevents
two concurrent promotions. Publication uses a same-directory atomic replace
and preserves the previous artifact as `.bak`. The evaluation ledger is
persisted before publication; failure to persist it blocks replacement.

The gate checks record identities and timestamp relationships. An exported
JSON file by itself cannot prove that its timestamps were not fabricated.
The immutable, pregame prediction recorder and automatic forward evaluation
belong to step 4; existing candidate snapshots cannot be relabeled as
prospective records. Raw game logs lack archived sportsbook prices, so their
rate holdout scores cannot bypass the price gate.

## Commands and evidence format

Training commands continue to stage candidates. Manifests are logged with
their paths. After genuinely prospective evidence has been recorded:

```powershell
python main.py promote --manifest <candidate-manifest.json> --observations <paired-predictions.json>
```

The evidence file is a JSON array. Each non-push row needs `candidate_id`,
`incumbent_id`, `event_id`, `game_date`, `player_name`, `market`, `line`, `side`,
`predicted_at`, `commence_time`, `settled_at`, `odds`, `candidate_prob`,
`incumbent_prob` and `won`. IDs are the SHA-256 values from the manifest;
timestamps must include timezone offsets. `push: true` rows are excluded.

Calibration is cached in the running scheduler: a promoted file is picked up
by its next scheduled reload or a restart. ML artifacts are read at inference.
No running service was restarted or production file promoted here.

## Verification

Tests exercise game/date isolation, target exclusion, removal of unarchived
features, forward stacking without final refitting, candidate persistence,
missing metadata, prospective price requirements, both baselines, tampered
files, duplicate rows, date/timestamp overlap, consumed-cohort rejection,
later-cohort acceptance and publication failures. Calibration tests confirm
staging without overwriting active files or fitting on holdout outcomes.

Tests use synthetic data and temporary artifacts. CLI argument wiring was
checked with `promote --help`. No live training/API fetches, Telegram messages,
production database writes or model promotions were used for validation.

Final test result: **280 passed; the same two pre-existing institutional
live-model tests failed** (portfolio rounding and enriched-versus-basic win
probability). Real XGBoost and ensemble fits were also exercised on small
synthetic datasets in temporary directories, including serialization and
confirmation that the active artifact path remained untouched.
