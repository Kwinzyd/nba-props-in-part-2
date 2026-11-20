"""Final validity checks for executable prop recommendations."""
from datetime import datetime, timezone
import math

from dateutil.parser import isoparse

from src.config import ALERT_LEAD_HOURS, MAX_PRICE_AGE_SECONDS
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def utc_time(value):
    try:
        dt = value if isinstance(value, datetime) else isoparse(value)
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (TypeError, ValueError, OverflowError):
        return None


def value_rejection(candidate):
    try:
        push = float(candidate.get('prob_push', 0))
    except (TypeError, ValueError):
        return 'invalid_numbers'
    if not math.isfinite(push) or not 0 <= push <= 1:
        return 'invalid_numbers'
    if push >= 1 - 1e-12:
        return 'certain_push'
    try:
        probability, odds, line = (float(candidate[k]) for k in ('model_prob', 'odds', 'line'))
        if not all(math.isfinite(v) for v in (probability, odds, line)):
            return 'invalid_numbers'
        if not 0 < probability < 1 or odds <= 1 or line < 0:
            return 'invalid_numbers'
    except (KeyError, ValueError, TypeError, OverflowError):
        return 'invalid_numbers'
    if probability * odds <= 1:
        return 'nonpositive_ev'
    return None


def timing_rejection(candidate, now=None):
    now = now or datetime.now(timezone.utc)
    tip = utc_time(candidate.get('commence_time'))
    if tip is None:
        return 'unknown_tipoff'
    seconds = (tip - now).total_seconds()
    if seconds <= 0:
        return 'game_started'
    if seconds > ALERT_LEAD_HOURS * 3600:
        return 'outside_alert_window'
    return None


def quote_rejection(candidate, quote, now=None):
    now = now or datetime.now(timezone.utc)
    if not quote:
        return 'quote_unavailable'
    if not math.isfinite(MAX_PRICE_AGE_SECONDS) or MAX_PRICE_AGE_SECONDS <= 0:
        return 'invalid_freshness_limit'
    # checked_at is the uncached response receipt time; use the upstream
    # update time as an additional constraint when the feed supplies one.
    for key in ('checked_at', 'updated_at'):
        if key == 'updated_at' and not quote.get(key):
            continue
        stamp = utc_time(quote.get(key))
        if stamp is None:
            return 'unknown_quote_time'
        age = (now - stamp).total_seconds()
        if age < -5 or age > MAX_PRICE_AGE_SECONDS:
            return 'stale_quote'
    try:
        if not math.isfinite(float(quote['price'])):
            return 'invalid_quote'
        if str(quote['book']).lower() != str(candidate['book']).lower():
            return 'quote_changed'
        if not math.isclose(float(quote['price']), float(candidate['odds']), rel_tol=0, abs_tol=1e-9):
            return 'quote_changed'
    except (KeyError, TypeError, ValueError):
        return 'invalid_quote'
    return None


def reject(candidate, db, reason):
    """Diagnostics are separate from alerts, results and calibration inputs."""
    candidate['selection_rejection'] = reason
    logger.info('Prop rejected: %s %s %s (%s)', candidate.get('player_id'),
                candidate.get('market'), candidate.get('side'), reason)
    try:
        with db.get_conn() as conn:
            conn.execute(
                'INSERT INTO selection_rejections (event_id, player_name, market, side, book, reason) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (candidate.get('event_id'), candidate.get('player_id'), candidate.get('market'),
                 candidate.get('side'), candidate.get('book'), reason))
    except Exception as exc:
        logger.warning('Could not persist selection rejection: %s', type(exc).__name__)
