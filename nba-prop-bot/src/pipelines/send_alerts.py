from math import prod, isfinite
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from src.utils.logging_utils import get_logger
from src.clients.telegram_bot import TelegramBotClient
from src.data.db import DatabaseClient, get_db_client
from src.clients.sportradar_client import SportradarClient
from src.config import (
    BANKROLL, KELLY_FRACTION, ELIMINATION_MODE, ELIMINATION_KELLY_MULT, GAME_MARKETS_ENABLED,
    DAILY_RISK_PCT, GAME_RISK_PCT, MAX_BET_PCT, STAKE_MODE, FLAT_STAKE_PCT, MAX_PRICE_AGE_SECONDS,
)
from src.execution.executor import record_execution
from src.pipelines.selection_guard import value_rejection, timing_rejection, quote_rejection, reject, utc_time

logger = get_logger(__name__)

# Compact market labels used in digest one-liners.
_MARKET_SHORT: Dict[str, str] = {
    'player_points':                  'Pts',
    'player_rebounds':                'Reb',
    'player_assists':                 'Ast',
    'player_threes':                  '3PM',
    'player_points_rebounds_assists': 'PRA',
    'player_blocks':                  'Blk',
    'player_steals':                  'Stl',
}


_CROSS_GAME_KELLY_MULTIPLIER = 0.5   # cross-game parlays have no structural edge

# Elimination games: stakes cut across the board (garbage-time variance).
_ELIM_KELLY_MULT = ELIMINATION_KELLY_MULT if ELIMINATION_MODE else 1.0


def _parlay_kelly_stake(
    legs: List[Dict[str, Any]],
    joint_true_prob: float,
    combined_decimal: float,
    max_pct: float = 0.005,
) -> float:
    """
    Kelly-sized stake for a multi-leg parlay.

    SGPs (all legs share one event_id) may have a Gaussian Copula edge baked in,
    so they use the full KELLY_FRACTION.  Cross-game parlays have no hidden
    correlation value — the book prices independent legs correctly — so their
    Kelly fraction is halved to account for the higher variance with no structural
    offset.
    """
    ev = joint_true_prob * (combined_decimal - 1) - (1 - joint_true_prob)
    if ev <= 0 or combined_decimal <= 1:
        return 0.0

    event_ids = {leg.get('event_id', '') for leg in legs if leg.get('event_id')}
    is_sgp = len(event_ids) <= 1
    effective_kelly = KELLY_FRACTION if is_sgp else KELLY_FRACTION * _CROSS_GAME_KELLY_MULTIPLIER
    effective_kelly *= _ELIM_KELLY_MULT

    stake = BANKROLL * (ev / (combined_decimal - 1)) * effective_kelly
    stake = min(stake, BANKROLL * max_pct)
    return min(_camouflage_stake(stake), stake)


def _camouflage_stake(stake: float) -> float:
    """Round to natural recreational-bettor increments to avoid sharp-profiling flags."""
    if stake <= 0:
        return 0.0
    if stake < 50:
        return max(5.0, round(stake / 5) * 5)
    if stake < 200:
        return round(stake / 25) * 25
    return round(stake / 50) * 50


def evaluate_and_alert(edge_data: Dict[str, Any], db: DatabaseClient, _bot: TelegramBotClient,
                       game_risk_cap: Optional[float] = None,
                       daily_risk_cap: Optional[float] = None, refresh_quote=None):
    """game_risk_cap / daily_risk_cap: tighter ceilings for this call (the
    scan's per-game fair share, and the daily budget minus shares reserved
    for later games); Kelly sizing still scales against MAX_PER_GAME."""
    reason = value_rejection(edge_data) or timing_rejection(edge_data)
    if reason:
        reject(edge_data, db, reason)
        return
    for key in ('model_prob', 'odds', 'line'):
        edge_data[key] = float(edge_data[key])
    # Recompute from the price/probability pair. Never trust a cached EV field.
    edge_data['ev'] = float(edge_data['model_prob']) * float(edge_data['odds']) - 1.0
    player    = edge_data.get('player_id', 'Unknown')
    market    = edge_data.get('market',    'Unknown')
    line      = edge_data.get('line',      0.0)
    side      = edge_data.get('side',      'Unknown')
    book      = edge_data.get('book',      'Unknown')
    edge      = edge_data.get('edge',      0.0)
    odds      = edge_data.get('odds',      0.0)
    game_date = edge_data.get('game_date', None)
    event_id  = edge_data.get('event_id',  None)
    home_away = edge_data.get('home_away', None)
    rest_days = edge_data.get('rest_days', 2)
    raw_model_prob = edge_data.get('raw_model_prob', None)
    model_prob_val = edge_data.get('model_prob', None)

    if db.check_recent_alert(player, market, line, side, edge):
        logger.info(f"Skipping duplicate alert for {player} {market} {side} {line}")
        return

    home = edge_data.get('home_team', 'Home')
    away = edge_data.get('away_team', 'Away')

    MAX_DAILY_RISK = BANKROLL * DAILY_RISK_PCT
    MAX_PER_GAME   = BANKROLL * GAME_RISK_PCT
    if not isfinite(MAX_PER_GAME) or MAX_PER_GAME <= 0:
        reject(edge_data, db, 'invalid_risk_limit')
        return

    current_daily_risk = 0.0
    current_game_risk = 0.0

    with db.get_conn() as conn:
        cursor = conn.cursor()

        # Daily total risk check
        cutoff = datetime.now(timezone.utc) - timedelta(hours=12)
        cursor.execute(
            "SELECT SUM(stake) as total_risk FROM alerts_sent WHERE timestamp >= ?", (cutoff,)
        )
        row = cursor.fetchone()
        current_daily_risk = float(row['total_risk'] or 0.0) if row else 0.0

        # Per-game risk check — group correlated bets within the same event
        if event_id:
            cursor.execute(
                "SELECT SUM(stake) as game_risk FROM alerts_sent "
                "WHERE event_id = ? AND timestamp >= ?",
                (event_id, cutoff),
            )
            game_row = cursor.fetchone()
            current_game_risk = float(game_row['game_risk'] or 0.0) if game_row else 0.0

    # ── Dynamic Kelly Sizing ──────────────────────────────────────────────────
    # 1. Covariance Multiplier: linearly decays Kelly as game exposure approaches MAX_PER_GAME.
    # e.g., if exposure is 0%, multiplier is 1.0. If exposure is 50% of cap, multiplier is 0.5.
    game_exposure_pct = current_game_risk / MAX_PER_GAME
    covariance_multiplier = max(0.1, 1.0 - game_exposure_pct)

    # 2. Isotonic Confidence Multiplier: tightens stakes in the high-variance 45-60% zone
    model_prob = edge_data.get('model_prob', 0.5)
    if 0.45 <= model_prob <= 0.60:
        isotonic_confidence_multiplier = 0.7
    else:
        isotonic_confidence_multiplier = 1.0

    dynamic_kelly = KELLY_FRACTION * covariance_multiplier * isotonic_confidence_multiplier * _ELIM_KELLY_MULT

    # Kelly formula: f* = ev / (odds - 1)
    ev    = edge_data.get('ev', 0.0)
    stake = 0.0
    if odds > 1 and ev > 0:
        if STAKE_MODE == 'flat':
            stake = BANKROLL * FLAT_STAKE_PCT * _ELIM_KELLY_MULT
        else:
            stake = BANKROLL * (ev / (odds - 1.0)) * dynamic_kelly
        stake = min(stake, BANKROLL * MAX_BET_PCT)  # hard cap per bet
    stake = min(_camouflage_stake(stake), stake)

    if not isfinite(stake) or stake <= 0:
        reject(edge_data, db, 'nonpositive_stake')
        return
    if ((daily_risk_cap is not None and (not isfinite(daily_risk_cap) or daily_risk_cap <= 0))
            or (game_risk_cap is not None and (not isfinite(game_risk_cap) or game_risk_cap <= 0))):
        reject(edge_data, db, 'risk_limit')
        return

    # Refresh only after cheap validity/deduplication/sizing checks. If the
    # price changes, defer to a later scan so calibration/ranking use that price.
    try:
        quote = refresh_quote(edge_data) if refresh_quote else None
    except Exception:
        quote = None
    reason = timing_rejection(edge_data) or quote_rejection(edge_data, quote)
    if reason:
        reject(edge_data, db, reason)
        return

    # ── Final Risk Guard & Persist ────────────────────────────────────────────
    expires_at = min(utc_time(edge_data['commence_time']),
                     utc_time(quote['checked_at']) + timedelta(seconds=MAX_PRICE_AGE_SECONDS))
    if quote.get('updated_at'):
        expires_at = min(expires_at, utc_time(quote['updated_at']) + timedelta(seconds=MAX_PRICE_AGE_SECONDS))
    # Atomic check-and-insert avoids double-spending risk limit in concurrent threads.
    alert_id = db.insert_alert_with_limits(
        player_name=player,
        market=market,
        line=line,
        side=side,
        edge=edge,
        book=book,
        odds=odds,
        stake=stake,
        game_date=game_date,
        event_id=event_id,
        home_away=home_away,
        rest_days=rest_days,
        max_daily_risk=min(MAX_DAILY_RISK, daily_risk_cap) if daily_risk_cap is not None else MAX_DAILY_RISK,
        max_per_game_risk=min(MAX_PER_GAME, game_risk_cap) if game_risk_cap is not None else MAX_PER_GAME,
        raw_model_prob=raw_model_prob,
        model_prob=model_prob_val,
        expires_at=expires_at,
    )

    if not alert_id:
        reject(edge_data, db, timing_rejection(edge_data) or quote_rejection(edge_data, quote) or 'risk_limit')
        logger.warning(f"Skipping {player} — daily or per-game risk limit reached.")
        return

    record_execution(alert_id, edge_data, stake, odds, db, _bot)

    ml_blend_note = " [ML+Bayesian blend]" if edge_data.get('ml_blend') else ""
    msg = (
        f"<b>🔥 NBA PROP EDGE</b>{ml_blend_note}\n\n"
        f"Game: {away} @ {home}  ({game_date})\n"
        f"Player: {player}  [{home_away}, {rest_days}d rest]\n"
        f"Market: {market.replace('_', ' ').title()}\n"
        f"Side: {side}  |  Line: {line}\n"
        f"Best Book: {book}  |  Odds: {odds}\n\n"
        f"Model Prob:   {edge_data.get('model_prob',   0.0):.3f}\n"
        f"Implied Prob: {edge_data.get('implied_prob', 0.0):.3f}\n"
        f"Edge:  {edge:.3%}\n"
        f"EV:    {edge_data.get('ev', 0.0):.3%}\n\n"
        f"Proj Mean:    {edge_data.get('mean',               0.0):.2f}\n"
        f"Proj Minutes: {edge_data.get('projected_minutes',  0.0):.1f}\n"
        f"Injury:       {edge_data.get('injury_status', 'Healthy')}\n"
        f"Usage Boost:  {edge_data.get('usage_boost',    0.0):.1%}\n"
        + (
            f"Sharp fair:   {edge_data['sharp_fair_prob']:.3f} "
            f"(break-even {1.0 / float(edge_data.get('odds') or 2.0):.3f})\n"
            if edge_data.get('sharp_fair_prob') is not None else ""
        )
        + (
            f"⚡ Line Shift: {edge_data['line_shift_old_line']} → "
            f"{edge_data['line_shift_new_line']} "
            f"@ {edge_data['line_shift_sharp_book']}\n"
            if edge_data.get('line_shift_flag') else ""
        )
        + f"\n<b>Suggested Stake ({'Flat' if STAKE_MODE == 'flat' else 'Kelly'}):</b> ${stake:.2f}"
    )

    # Attempt to attach Sportradar Logo
    # This requires a Sportradar player ID, which we may look up or receive via edge_data
    sr_player_id = edge_data.get('sr_player_id')
    if sr_player_id:
        sr_client = SportradarClient()
        logo_url = sr_client.get_player_logo_url(sr_player_id)
        # Add an invisible link to force Telegram link preview of the image
        msg = f"<a href='{logo_url}'>&#8205;</a>" + msg

    # Send immediately to all subscribers so they get the edge before the line moves.
    # A failed send lands in the Telegram outbox and is retried (flush_outbox).
    delivered = 0
    try:
        delivered = _bot.broadcast(msg, db=db, expires_at=expires_at.timestamp())
    except Exception as _e:
        logger.warning(f"Telegram broadcast failed (will still queue): {_e}")

    # Prop prices expire quickly. Scheduled digests cannot revalidate these
    # text-only recommendations, so deliver through the expiring instant path.
    logger.info(
        f"Alert {'sent' if delivered else 'NOT delivered'}: "
        f"{player} {market} {side} {line} @ {book}")


def _claim_line_disagreement(db, game_date, event_id, player, market, soft_book,
                             soft_line, side, soft_odds, sharp_book, sharp_line) -> bool:
    """Record the day's line-disagreement alert for a player. False if one was already sent."""
    with db.get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO line_disagreement_alerts
                (game_date, event_id, player_name, market, book, line, side, odds,
                 sharp_book, sharp_line)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (game_date, player_name) DO NOTHING
            """,
            (game_date, event_id, player, market, soft_book, soft_line, side,
             soft_odds, sharp_book, sharp_line),
        )
        return cur.rowcount > 0


def send_line_disagreement_alert(
    player: str, market: str, sharp_book: str, sharp_line: float,
    soft_book: str, soft_line: float, side: str, soft_odds: float,
    game_date: str, event_id: str, home_team: str, away_team: str,
    db: DatabaseClient, _bot: TelegramBotClient,
):
    """
    Fire a priority alert when a sharp book's line differs from a soft book
    by >= 1 full point.  These are the highest-ROI bets in the market — the
    soft book hasn't adjusted yet and is offering a line the sharp book has
    already moved past.
    """
    line_gap = abs(sharp_line - soft_line)

    # One line-disagreement alert per player per day, across scans and
    # restarts. (These were checked against alerts_sent, which they are never
    # written to, so every scan re-sent them.)
    if not _claim_line_disagreement(db, game_date, event_id, player, market,
                                    soft_book, soft_line, side, soft_odds,
                                    sharp_book, sharp_line):
        logger.debug(f"Line disagreement already sent today for {player}")
        return

    mkt_short = _MARKET_SHORT.get(market, market[:3].title())

    msg = (
        f"<b>⚡ SHARP LINE DISAGREEMENT</b>\n\n"
        f"Game: {away_team} @ {home_team}  ({game_date})\n"
        f"Player: {player}\n"
        f"Market: {market.replace('_', ' ').title()}\n\n"
        f"<b>{sharp_book}</b> line: {sharp_line}\n"
        f"<b>{soft_book}</b> line:  {soft_line}  ({line_gap:+.1f} gap)\n\n"
        f"Action: <b>{side} {soft_line}</b> on {soft_book}\n"
        f"Odds: {soft_odds}\n\n"
        f"A {line_gap:.1f}-point line gap on a Poisson distribution is massive.\n"
        f"Hammer this before {soft_book} adjusts."
    )

    title = (
        f"⚡ {player} {side[0]}{soft_line} {mkt_short} "
        f"@{soft_book} | {sharp_book} @ {sharp_line} ({line_gap:+.1f})"
    )

    # Send immediately — line disagreements are highest urgency (soft book adjusts fast).
    try:
        _bot.broadcast(msg, db=db)
    except Exception as _e:
        logger.warning(f"Telegram broadcast failed (will still queue): {_e}")

    # Queue as highest-priority alert (priority = line_gap to sort above normal edges)
    db.queue_pending_alert('prop', title, msg, priority=line_gap, game_date=game_date)
    logger.info(
        f"LINE DISAGREEMENT sent + queued: {player} {market} — {sharp_book} {sharp_line} vs "
        f"{soft_book} {soft_line} ({side})"
    )


_PARLAY_MARKET_LABELS: Dict[str, str] = {
    'player_points':                  'Points',
    'player_rebounds':                'Rebounds',
    'player_assists':                 'Assists',
    'player_threes':                  'Threes',
    'player_points_rebounds_assists': 'PRA',
    'player_blocks':                  'Blocks',
    'player_steals':                  'Steals',
}


def _parlay_american(dec: float) -> str:
    if dec >= 2.0:
        return f"+{int((dec - 1) * 100)}"
    if dec <= 1.0:
        return "N/A"
    return f"-{int(100 / (dec - 1))}"


def send_parlay_alert(
    legs: List[Dict[str, Any]],
    joint_true_prob: float,
    joint_book_prob: float,
    bot: TelegramBotClient,
    db: Optional[DatabaseClient] = None,
) -> None:
    """
    Format and send a multi-leg high-probability parlay to Telegram.

    Displays each leg with its individual hit probability and game context,
    then shows combined book odds, true hit probability, edge, and EV.

    Args:
        legs:             List of edge dicts (player_id, market, side, line,
                          book, odds, model_prob, home_team, away_team).
        joint_true_prob:  Model's joint probability all legs hit.
        joint_book_prob:  Book's implied joint probability (product of vigs).
        bot:              Telegram client.
    """
    n = len(legs)
    icon = '4️⃣' if n == 4 else '8️⃣'

    combined_decimal = prod(leg['odds'] for leg in legs)
    edge = joint_true_prob - joint_book_prob
    ev   = joint_true_prob * (combined_decimal - 1) - (1 - joint_true_prob)

    _PARLAY_MAX_PCT = {4: 0.005, 8: 0.0025}
    max_pct = _PARLAY_MAX_PCT.get(n, 0.0025)
    kelly_stake = _parlay_kelly_stake(legs, joint_true_prob, combined_decimal, max_pct=max_pct)

    lines = [f"{icon} <b>{n}-Leg High-Probability Parlay</b>\n"]
    for leg in legs:
        market = _PARLAY_MARKET_LABELS.get(leg.get('market', ''), leg.get('market', ''))
        away   = leg.get('away_team', '')
        home   = leg.get('home_team', '')
        game   = f"[{away} @ {home}]" if away and home else ""
        prob   = leg.get('model_prob', 0.0)
        lines.append(
            f"• <b>{leg.get('player_id', '?')}</b> "
            f"{leg.get('side', '')} {leg.get('line', '')} {market} "
            f"@ {leg.get('book', '')} ({_parlay_american(leg.get('odds', 1.0))}) "
            f"{game} — {prob:.0%}"
        )

    # Thresholds for comparing against the book's actual SGP price:
    # BET if book offers ≥ 85% of theoretical baseline (≤15% SGP penalty)
    # SKIP if book offers < 60% of theoretical baseline (>40% SGP penalty)
    bet_threshold  = _parlay_american(combined_decimal * 0.85)
    skip_threshold = _parlay_american(combined_decimal * 0.60)

    lines.append(
        f"\nTheoretical Baseline: {_parlay_american(combined_decimal)}\n"
        f"⚠️ Books apply hidden SGP penalties — verify the book's actual SGP price.\n"
        f"✅ BET if book offers ≥ {bet_threshold} | ❌ SKIP if book offers < {skip_threshold}\n"
        f"True Hit Prob:  {joint_true_prob:.2%}\n"
        f"Book Implied:   {joint_book_prob:.2%}\n"
        f"Edge:           +{edge:.2%}\n"
        f"EV:             {ev:+.2%}\n"
        f"Suggested Stake (Kelly): ${kelly_stake:.0f} (cap: {max_pct:.2%} bankroll)"
    )

    msg = "\n".join(lines)
    if db is not None:
        # Tier 2: queue for digest.
        title = f"{n}-leg SGP | True: {joint_true_prob:.0%} | Edge: {edge:+.1%} | EV: {ev:+.1%}"
        db.queue_pending_alert('parlay', title, msg, priority=ev)
    else:
        # Fallback: send instantly when no DB is available.
        bot.send_message(msg)
    logger.info(
        f"{n}-leg high-prob parlay | "
        f"joint_prob={joint_true_prob:.2%} | edge={edge:.2%} | ev={ev:.2%}"
    )


# ── Game market alerts (Moneyline / Spread / Total) ───────────────────────────

_GAME_MARKET_LABELS: Dict[str, str] = {
    'h2h':     'Moneyline',
    'spreads': 'Spread',
    'totals':  'Total',
}


def send_game_market_alert(
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
    market: str,
    side: str,
    edge: float,
    ev: float,
    model_prob: float,
    book_prob: float,
    book_odds: float,
    book: str,
    game_date: str,
    event_id: str,
    line: float,
    db: DatabaseClient,
    _bot: TelegramBotClient,
    home_abbr: str = '',
    away_abbr: str = '',
) -> None:
    """
    Evaluate and send a Telegram alert for a game market edge
    (Moneyline, Spread, or Total).

    `side` doubles as both the DB deduplication key and the display label
    (e.g. "Denver Nuggets", "Boston Celtics -6.5", "Over").
    `line` is 0.0 for moneyline, the spread value for spreads, and the
    combined total for totals.
    """
    if not GAME_MARKETS_ENABLED:
        return
    matchup      = f"{away_team} @ {home_team}"
    market_label = _GAME_MARKET_LABELS.get(market, market.title())

    if db.check_recent_alert(matchup, market, line, side, edge):
        logger.info(f"Skipping duplicate game market alert: {matchup} {market_label} {side}")
        return

    # Kelly sizing — capped at 3 % of bankroll (game markets are more efficient
    # than player props; model uncertainty is higher for team-level projections).
    stake = 0.0
    if book_odds > 1.0 and ev > 0.0:
        stake = BANKROLL * (ev / (book_odds - 1.0)) * KELLY_FRACTION
        stake = min(stake, BANKROLL * MAX_BET_PCT)
    stake = min(_camouflage_stake(stake), stake)

    MAX_DAILY_RISK = BANKROLL * DAILY_RISK_PCT
    MAX_PER_GAME   = BANKROLL * GAME_RISK_PCT

    with db.get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT SUM(stake) as total_risk FROM alerts_sent WHERE timestamp >= NOW() - INTERVAL '12 hours'"
        )
        row = cursor.fetchone()
        current_daily_risk = float(row['total_risk'] or 0.0) if row else 0.0
        if current_daily_risk + stake > MAX_DAILY_RISK:
            logger.warning(
                f"Skipping {matchup} — daily risk limit "
                f"({current_daily_risk:.2f}/{MAX_DAILY_RISK:.2f})"
            )
            return

        if event_id:
            cursor.execute(
                "SELECT SUM(stake) as game_risk FROM alerts_sent "
                "WHERE event_id = ? AND timestamp >= NOW() - INTERVAL '12 hours'",
                (event_id,),
            )
            game_row = cursor.fetchone()
            current_game_risk = float(game_row['game_risk'] or 0.0) if game_row else 0.0
            if current_game_risk + stake > MAX_PER_GAME:
                logger.warning(
                    f"Skipping {matchup} — per-game risk limit "
                    f"({current_game_risk:.2f}/{MAX_PER_GAME:.2f})"
                )
                return

    alert_id = db.insert_alert(
        player_name=matchup, market=market, line=line, side=side,
        edge=edge, book=book, odds=book_odds, stake=stake,
        game_date=game_date, event_id=event_id,
        raw_model_prob=model_prob, model_prob=model_prob,
    )
    game_edge_data = {
        'player_name': matchup, 'market': market, 'side': side, 'line': line,
        'book': book, 'game_date': game_date,
    }
    record_execution(alert_id, game_edge_data, stake, book_odds, db, _bot)

    _h_abbr = home_abbr or home_team.split()[-1][:3].upper()
    _a_abbr = away_abbr or away_team.split()[-1][:3].upper()
    line_str = f" {line:+.1f}" if market != 'h2h' else ''

    msg = (
        f"<b>⚡ NBA GAME MARKET EDGE</b>\n\n"
        f"Game: {_a_abbr} @ {_h_abbr}  ({game_date})\n"
        f"Proj: {_a_abbr} {away_score:.1f} – {_h_abbr} {home_score:.1f}\n\n"
        f"Market: {market_label}{line_str}\n"
        f"Side:   {side}\n"
        f"Best Book: {book}  |  Odds: {_parlay_american(book_odds)}\n\n"
        f"Model Prob:  {model_prob:.3f}\n"
        f"Book Prob:   {book_prob:.3f}\n"
        f"Edge:  {edge:+.2%}\n"
        f"EV:    {ev:+.2%}\n\n"
        f"<b>Suggested Stake (Kelly):</b> ${stake:.0f}"
    )

    # Send immediately so subscribers can act before the line moves.
    try:
        _bot.broadcast(msg, db=db)
    except Exception as _e:
        logger.warning(f"Telegram broadcast failed (will still queue): {_e}")

    # Tier 2: queue for digest.
    side_short = side if len(side) <= 14 else side[:13] + "…"
    title = (
        f"{_a_abbr}@{_h_abbr} {market_label}{line_str} {side_short} "
        f"@{book} {_parlay_american(book_odds)} | {edge:+.1%} | ${stake:.0f}"
    )
    db.queue_pending_alert('game_market', title, msg, priority=edge, game_date=game_date)
    logger.info(f"Game market alert sent + queued: {matchup} {market_label} {side} edge={edge:.2%}")
