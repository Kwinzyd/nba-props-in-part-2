"""
CLV Closing Line Capture.

For each unsettled CLV tracker whose game is about to tip, fetches the
Pinnacle / Circa / Bookmaker closing price and saves the CLV delta.

  • Event-targeted  — one API call per event (not one per player), only for
                      the markets that were alerted.
  • Once per event  — the first run inside the closing window fetches the
                      line; later runs skip the event so credits aren't
                      re-spent on trackers with no sharp price.
  • Pinnacle-first  — sharp books in priority order; Pinnacle is the gold
                      standard closing line signal.
  • Same-book fallback — sharp books don't post steals/blocks and often hang
                      a different line, so without a sharp price we compare
                      against the alerted book's own close (same response, no
                      extra credits). clv_tracking.close_book records which.
  • Same line only  — a closing price is only comparable when the sharp
                      book still hangs the alerted line.
  • JOIN enrichment — joins clv_tracking with alerts_sent to retrieve the
                      event_id and line without a schema change.

Credits per run: 0 (get_events is free) + 1 per alerted market per tipping
event, charged to the separate ODDS_CLV_CREDIT_BUDGET allowance.

Schedule: every 5 minutes on game days (see run_scheduler.py), so each game
gets a run inside the closing window.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from src.data.db import get_db_client
from src.clients.odds_api import OddsApiClient, OddsApiQuotaError, OddsApiBudgetError
from src.clients.sportradar_client import SportradarClient
from src.models.devig import decimal_to_implied_prob
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ── Timing window ─────────────────────────────────────────────────────────────
# Capture closing lines from 12 minutes before scheduled tip-off until 10
# minutes after (sharp books pull props once the game is live; the tail only
# covers a run delayed by a busy job pool).
_WINDOW_FUTURE = timedelta(minutes=12)   # how far ahead of tip-off we may grab
_WINDOW_PAST   = timedelta(minutes=10)   # how far past tip-off we still try

# Trackers older than this belong to games long settled (or to old seasons).
_TRACKER_MAX_AGE = '2 days'

# Sharp books in priority order (Odds API bookmaker keys).  Pinnacle is the
# only globally-accepted closing-line benchmark; the rest are fallbacks.
_SHARP_PRIORITY = ['pinnacle', 'circasports', 'bookmaker', 'betonlineag']

# Events whose closing line was already fetched by this process.
_fetched_events: set = set()


def _norm(s: str) -> str:
    """Lowercase + strip for matching player names / side labels."""
    return s.strip().lower()


def update_clv_lines() -> None:
    """
    Snapshot closing lines for all unsettled CLV trackers in the tip-off window.
    """
    db          = get_db_client()
    odds_client = OddsApiClient()

    # ── 1. Pull recent unsettled trackers + event context ─────────────────
    # JOIN with alerts_sent so we can group by event_id without a schema change.
    with db.get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT ct.id              AS track_id,
                   ct.player_id       AS player,
                   ct.market,
                   ct.side,
                   ct.alert_odds,
                   a.event_id,
                   a.line,
                   a.book,
                   a.game_date
            FROM   clv_tracking ct
            LEFT   JOIN alerts_sent a ON a.id = ct.alert_id
            WHERE  ct.closing_odds IS NULL
              AND  ct.alert_time >= NOW() - INTERVAL '{_TRACKER_MAX_AGE}'
            ORDER  BY ct.alert_time
            """
        ).fetchall()

    if not rows:
        logger.debug("CLV: no unsettled trackers.")
        return

    # ── 2. Group by event_id ──────────────────────────────────────────────
    by_event: Dict[str, List[dict]] = {}
    for row in rows:
        eid = row['event_id']
        if eid and eid not in _fetched_events:
            by_event.setdefault(eid, []).append(dict(row))
        elif not eid:
            logger.debug(f"CLV: no event_id for {row['player']} {row['market']} — skipping.")

    if not by_event:
        logger.debug("CLV: no trackers awaiting a closing line.")
        return

    # ── 3. Fetch upcoming events to check commence times (free) ───────────
    try:
        events_list = odds_client.get_events()
    except Exception as exc:
        logger.error(f"CLV: get_events failed: {exc}")
        return

    commence_map: Dict[str, datetime] = {}
    for ev in events_list:
        try:
            dt = datetime.fromisoformat(ev['commence_time'].replace('Z', '+00:00'))
            commence_map[ev['id']] = dt
        except Exception:
            pass

    now     = datetime.now(timezone.utc)
    updated = 0

    # ── 4. Process each event in the tip-off window ───────────────────────
    for event_id, trackers in by_event.items():
        commence = commence_map.get(event_id)

        if commence is None:
            # Event not in upcoming list — game is likely already over.
            logger.debug(f"CLV: event {event_id} not in upcoming events — skipping.")
            continue

        age = now - commence  # positive = game started; negative = in future

        if age < -_WINDOW_FUTURE:
            # Too far in future — market still moving; wait for closing line
            logger.debug(
                f"CLV: event {event_id} tips in "
                f"{-age.total_seconds()/60:.0f}min — skipping."
            )
            continue

        if age > _WINDOW_PAST:
            # Closing window missed — sharp books have pulled props by now.
            logger.info(
                f"CLV: event {event_id} started "
                f"{age.total_seconds()/60:.0f}min ago — closing window missed."
            )
            _fetched_events.add(event_id)
            continue

        # One API call per event (not per player)
        markets_needed = sorted({t['market'] for t in trackers})
        try:
            if event_id.startswith("sr:"):
                logger.info(f"CLV: event_id {event_id} is a Sportradar ID. Fetching directly from Sportradar.")
                sr_client = SportradarClient()
                odds_data = sr_client.get_event_odds_oddsapi_format(event_id)
            else:
                try:
                    odds_data = odds_client.get_event_odds(
                        event_id=event_id,
                        markets=markets_needed,
                        caller='clv',
                        use_cache=False,   # closing line, not the scan's pre-game fetch
                    )
                except OddsApiBudgetError as exc:
                    logger.warning(f"CLV: {exc}")
                    _fetched_events.add(event_id)
                    continue
                except OddsApiQuotaError:
                    logger.warning(f"CLV: Odds API out of credits for event {event_id}. Falling back to Sportradar.")
                    with db.get_conn() as conn:
                        row = conn.execute("SELECT sr_id FROM games WHERE game_id = ?", (event_id,)).fetchone()
                        if row and row['sr_id']:
                            sr_client = SportradarClient()
                            odds_data = sr_client.get_event_odds_oddsapi_format(row['sr_id'])
                        else:
                            raise Exception("No sr_id available for fallback.")
        except Exception as exc:
            logger.warning(f"CLV: odds fetch failed for event {event_id}: {exc}")
            continue

        _fetched_events.add(event_id)
        bookmakers = odds_data.get('bookmakers', [])

        # Build lookup: {(book_key, market_key, player_lower, side_lower, point): price}
        prices: Dict[Tuple, float] = {}
        for book in bookmakers:
            # Alerts store either the key ('betmgm') or the title ('Caesars'
            # for key 'williamhill_us'), so index each book under both.
            book_ids = {_norm(book.get('key') or ''), _norm(book.get('title') or '')} - {''}
            for mkt in book.get('markets', []):
                mkt_key = mkt.get('key', '')
                for outcome in mkt.get('outcomes', []):
                    desc  = _norm(outcome.get('description', ''))  # player name
                    name  = _norm(outcome.get('name', ''))          # OVER/UNDER
                    price = float(outcome.get('price', 0) or 0)
                    point = outcome.get('point')
                    if price > 1.0 and point is not None:
                        for book_id in book_ids:
                            prices[(book_id, mkt_key, desc, name, float(point))] = price

        # ── Grade each tracker against the closing price ──────────────────
        for tracker in trackers:
            player_n   = _norm(tracker['player'])
            market     = tracker['market']
            side_n     = _norm(tracker['side'])
            track_id   = tracker['track_id']
            alert_odds = float(tracker['alert_odds'])
            line       = tracker['line']
            if line is None:
                continue

            # Find sharpest available closing price at the alerted line,
            # else the alerted book's own closing price.
            closing_price = 0.0
            source_book   = ''
            for book_id in _SHARP_PRIORITY + [_norm(tracker.get('book') or '')]:
                key = (book_id, market, player_n, side_n, float(line))
                if book_id and key in prices:
                    closing_price = prices[key]
                    source_book   = book_id
                    break

            if closing_price <= 1.0:
                logger.debug(
                    f"CLV: no sharp price for "
                    f"{tracker['player']} {market} {tracker['side']} {line}"
                )
                continue

            implied_closing = decimal_to_implied_prob(closing_price)
            implied_alert   = decimal_to_implied_prob(alert_odds)
            db.update_clv_closing_line(
                track_id, closing_price, implied_closing, implied_alert,
                close_book=source_book,
            )
            updated += 1
            logger.info(
                f"CLV: {tracker['player']} {market} {tracker['side']} {line} "
                f"alert={alert_odds:.3f} → close={closing_price:.3f} "
                f"({source_book}) | CLV={implied_closing - implied_alert:+.4f}"
            )

    logger.info(f"CLV update complete: {updated} closing line(s) recorded.")


if __name__ == "__main__":
    update_clv_lines()
