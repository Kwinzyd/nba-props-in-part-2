"""
Shadow overs: paper-track the overs the model would bet while OVERS_ENABLED
is off, so turning them back on can be decided from evidence.

Overs were disabled on 2026-10-27 (22-42, 34% hit vs 52% implied). The
market-anchored calibration now gives overs the market price (k=0), so the
live rules would select almost none — the open question is whether the
model's *own* over signal beats the price. A shadow pick is therefore an
over candidate re-ranked on its raw (pre-calibration) model probability that
clears the same edge floor, minutes/injury filters and ALERT_LEAD_HOURS
window as a real alert.

Picks live in shadow_picks (not alerts_sent), so they never touch risk caps,
dedup, calibration fits, CLV or P&L. First qualifying sighting per
(game_date, player, market, line, side) is kept, like a real alert. Graded in
the daily settlement run from the same BDL box scores. Report with
`python main.py shadow_stats`.
"""

from typing import Dict, List, Optional

from src.config import ALERT_LEAD_HOURS, OVERS_ENABLED, SHADOW_OVERS_ENABLED
from src.models.edge_ranker import rank_edges
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def select_shadow_overs(candidates: List[dict]) -> List[dict]:
    """Calibrated over candidates that clear the edge floor on their raw model prob."""
    raw_overs = []
    for c in candidates:
        if str(c.get('side', '')).upper() != 'OVER' or c.get('raw_model_prob') is None:
            continue
        shadow = dict(c)                       # rank_edges mutates in place
        shadow['model_prob_calibrated'] = c.get('model_prob')
        shadow['model_prob'] = c['raw_model_prob']
        raw_overs.append(shadow)
    return [
        e for e in rank_edges(raw_overs)
        if e.get('edge', 0) >= e.get('edge_min_applied', 1.0)
        and e.get('hours_to_tipoff', 0.0) <= ALERT_LEAD_HOURS
    ]


def record_shadow_overs(candidates: List[dict], db) -> int:
    """Store new shadow over picks from a scan's calibrated candidates. Returns # new."""
    if OVERS_ENABLED or not SHADOW_OVERS_ENABLED:
        return 0
    picks = select_shadow_overs(candidates)
    new = 0
    with db.get_conn() as conn:
        for p in picks:
            cur = conn.execute(
                """
                INSERT INTO shadow_picks
                    (game_date, event_id, player_name, market, line, side, book, odds,
                     implied_prob, raw_model_prob, model_prob, edge, edge_min, hours_to_tipoff)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (game_date, player_name, market, line, side) DO NOTHING
                """,
                (p.get('game_date'), p.get('event_id'), p.get('player_name'), p.get('market'),
                 p.get('line'), 'OVER', p.get('book'), p.get('odds'), p.get('implied_prob'),
                 p.get('raw_model_prob'), p.get('model_prob_calibrated'), p.get('edge'),
                 p.get('edge_min_applied'), p.get('hours_to_tipoff')),
            )
            new += max(cur.rowcount, 0)
    if new:
        logger.info(f"Shadow overs: {new} new paper pick(s) recorded ({len(picks)} qualified this scan).")
    return new


# Paper-pick tables sharing this grading/report code (same core columns).
PAPER_TABLES = {'shadow_picks': 'Shadow overs', 'sharp_picks': 'Sharp picks'}


def _check_table(table: str) -> str:
    if table not in PAPER_TABLES:
        raise ValueError(f"not a paper-pick table: {table}")
    return table


def unsettled_shadow_dates(db, start: str, end: str, table: str = 'shadow_picks') -> List[str]:
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT game_date FROM {_check_table(table)} "
            "WHERE settled_at IS NULL AND game_date BETWEEN ? AND ? ORDER BY 1",
            (start, end),
        ).fetchall()
    return [str(r['game_date']) for r in rows]


def settle_shadow_picks(db, target_date: str, find_stat_row, grade_prop,
                        table: str = 'shadow_picks') -> int:
    """
    Grade paper picks for target_date. find_stat_row(player_name) returns the
    player's final box-score row or None; grade_prop is settle_results'
    _grade_prop. Picks with no stat row (game not final) stay ungraded.
    """
    label = PAPER_TABLES[_check_table(table)]
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT id, player_name, market, line, side FROM {table} "
            "WHERE settled_at IS NULL AND game_date = ?",
            (target_date,),
        ).fetchall()
    graded = 0
    for r in rows:
        stat_row = find_stat_row(r['player_name'])
        if stat_row is None:
            continue
        won, actual = grade_prop(r['market'], float(r['line']), r['side'], stat_row)
        with db.get_conn() as conn:
            conn.execute(
                f"UPDATE {table} SET actual_result = ?, won = ?, push = ?, settled_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (actual, bool(won) if won is not None else False, won is None, r['id']),
            )
        graded += 1
    if rows:
        logger.info(f"{label}: graded {graded}/{len(rows)} paper pick(s) for {target_date}.")
    return graded


def shadow_report(db, since: Optional[str] = None, table: str = 'shadow_picks') -> Dict[str, dict]:
    """Print and return hit rate vs break-even and flat-stake ROI, overall and by market."""
    label = PAPER_TABLES[_check_table(table)]
    with db.get_conn() as conn:
        rows = conn.execute(
            f"SELECT market, odds, won, edge FROM {table} "
            "WHERE settled_at IS NOT NULL AND NOT push AND game_date >= ?",
            (since or '0000-00-00',),
        ).fetchall()
    groups: Dict[str, list] = {'ALL': list(rows)}
    for r in rows:
        groups.setdefault(r['market'], []).append(r)
    out = {}
    print(f"{label} (graded, excl. pushes and voids){' since ' + since if since else ''}:")
    for name, rs in groups.items():
        n = len(rs)
        if not n:
            print("  no graded picks yet")
            continue
        wins = sum(1 for r in rs if r['won'])
        implied = sum(1.0 / float(r['odds']) for r in rs) / n
        units = sum((float(r['odds']) - 1.0) if r['won'] else -1.0 for r in rs)
        se = (implied * (1 - implied) / n) ** 0.5
        out[name] = {'n': n, 'wins': wins, 'hit': wins / n, 'implied': implied,
                     'roi': units / n, 'z': (wins / n - implied) / se if se else 0.0}
        s = out[name]
        print(f"  {name:32s} {wins}-{n - wins}  hit {s['hit']:.1%} vs break-even {implied:.1%}"
              f"  (z={s['z']:+.1f})  ROI {s['roi']:+.1%}")
    return out
