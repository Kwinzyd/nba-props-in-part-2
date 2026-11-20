# Prop selection correctness patch

Prop recommendations now require finite valid prices/probabilities, positive
EV at the offered odds, a positive allowed stake, and a known timezone-aware
start time inside the alert window. The start time is checked again after
quote refresh; the database insert also checks the recommendation deadline.
PostgreSQL uses wall-clock time after acquiring the risk lock, so waiting for
another writer cannot extend a recommendation past its expiry.

Edge floors now apply to probability above **1 / offered odds**. The separate
`fair_market_edge` diagnostic retains comparison with devigged market prices.
Configured floors were not lowered. With the current calibration's k=0 and
b_under=0.1577, its maximum under lift is below the lowest 4% floor, so no
standard prop plays may qualify until validated model changes support them.

The sender confirms the exact player, market, book, line and side against an
uncached feed lookup. Changed or missing quotes are rejected until a later
scan can recalibrate/rank them. Responses are shared for at most 30 seconds
within a scan. The Odds API refresh bypasses its long cache but retains the
existing credit-budget guard; refusal means no recommendation.

`MAX_PRICE_AGE_SECONDS` defaults to 120. Both response observation and upstream
update timestamps are checked when the latter exists. BDL does not always
supply upstream timestamps: a fresh observation confirms the current feed
response, not the underlying sportsbook's update time or a guaranteed fill.

Rejections are stored in `selection_rejections`, separately from alerts, CLV,
results and calibration data. Existing candidate recording remains available.
Accepted prop alerts use expiring instant delivery; new props are no longer
copied into scheduled digests at potentially stale prices. Outbox retries obey
the earlier of quote expiry and tipoff. Older queued messages without expiry
metadata have not been rewritten. Disabled game-market/parlay strategies and
their dedicated senders are outside this prop-path patch.

Tests: 234 passed; two pre-existing institutional live-model tests still fail
(portfolio rounding and enriched-versus-basic win probability). New tests
exercise actual SQLite sender persistence, rejection isolation, successful
deduplicated delivery, price changes/missing quotes, invalid numeric inputs,
zero stake/risk, time boundaries, delayed expiry, and forced uncached lookups.
The PostgreSQL insert/deadline SQL passed read-only EXPLAIN against the active
database. No live provider calls or Telegram messages were sent for validation.

The scheduler has not been restarted. On restart, normal schema initialization
creates the diagnostics table and loads these guards. Fitted models and
historical recommendations remain unchanged by this patch.
