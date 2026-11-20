"""
Sharp picks: paper-track every Sharp Alert so the strategy can be judged.

A Sharp Alert fires when the best DraftKings/FanDuel price for a prop beats
the devigged sharp consensus (Pinnacle/Circa/Bookmaker) by SHARP_EDGE_MIN:
edge = fair_prob - 1/odds. Unlike the model alerts this needs no projection
— it is a pure price comparison — so it is the strategy to test first.

Until 2026-11-09 these alerts were only sent to Telegram, never stored, so
there was no record of how they did. Picks live in sharp_picks (not
alerts_sent): no stakes, no effect on risk caps, calibration, CLV or P&L.
First sighting per (game_date, player, market, line, side) is kept, and only
that first sighting is sent to Telegram. Graded in the daily settlement run;
report with `python main.py sharp_stats`.
"""

from typing import Optional

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def record_sharp_pick(db, *, game_date: str, event_id: Optional[str], player_name: str,
                      market: str, line: float, side: str, book: str, odds: float,
                      fair_prob: float, consensus_label: Optional[str],
                      hours_to_tipoff: Optional[float]) -> bool:
    """Store a sharp pick. True if new (first sighting), False if already recorded."""
    with db.get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO sharp_picks
                (game_date, event_id, player_name, market, line, side, book, odds,
                 fair_prob, consensus_label, edge, hours_to_tipoff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (game_date, player_name, market, line, side) DO NOTHING
            """,
            (game_date, event_id, player_name, market, line, side.upper(), book, odds,
             fair_prob, consensus_label, fair_prob - 1.0 / odds, hours_to_tipoff),
        )
        new = cur.rowcount > 0
    if new:
        logger.info(
            f"Sharp pick: {player_name} {market} {side.upper()} {line} @ {odds:.2f} {book} "
            f"| fair {fair_prob:.1%} vs break-even {1.0 / odds:.1%}"
        )
    return new
