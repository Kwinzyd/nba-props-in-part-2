"""
Candidate log: store every prop the scan evaluated, so the model can be
scored against the market on all props rather than the few it bet.

One row per (game_date, player, market, line, side), refreshed on every scan
until it is settled, so the stored row is the last pre-tip view: the latest
DK/FD price, the model's raw and calibrated probabilities, the projection,
and the devigged sharp consensus when the Odds API was fetched. Graded in the
daily settlement run (DNPs void); `python main.py model_eval` compares the
raw model with the market price.
"""

import math
from typing import Dict, List, Optional

from src.config import ALT_TO_BASE_MARKET, CANDIDATE_LOG_ENABLED
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

_UPSERT = """
INSERT INTO scan_candidates
    (game_date, event_id, player_name, market, line, side, book, odds, implied_prob,
     raw_model_prob, model_prob, projected_mean, projected_minutes, injury_status,
     sharp_over_prob, consensus_label, hours_to_tipoff, edge)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (game_date, player_name, market, line, side) DO UPDATE SET
    event_id = excluded.event_id, book = excluded.book, odds = excluded.odds,
    implied_prob = excluded.implied_prob, raw_model_prob = excluded.raw_model_prob,
    model_prob = excluded.model_prob, projected_mean = excluded.projected_mean,
    projected_minutes = excluded.projected_minutes, injury_status = excluded.injury_status,
    sharp_over_prob = COALESCE(excluded.sharp_over_prob, scan_candidates.sharp_over_prob),
    consensus_label = COALESCE(excluded.consensus_label, scan_candidates.consensus_label),
    hours_to_tipoff = excluded.hours_to_tipoff, edge = excluded.edge,
    updated_at = CURRENT_TIMESTAMP
WHERE scan_candidates.settled_at IS NULL
"""


def _row(c: dict) -> Optional[tuple]:
    player, odds = c.get('player_id'), c.get('odds') or 0.0
    if not player or odds <= 1.0 or not c.get('game_date'):
        return None
    market = ALT_TO_BASE_MARKET.get(c.get('market', ''), c.get('market', ''))
    raw = c.get('raw_model_prob', c.get('model_prob'))
    return (c['game_date'], c.get('event_id'), player, market, c.get('line'),
            str(c.get('side', '')).upper(), c.get('book'), odds, c.get('implied_prob'),
            raw, c.get('model_prob'), c.get('mean'), c.get('projected_minutes'),
            c.get('injury_status'), c.get('consensus_prob'), c.get('consensus_label'),
            c.get('hours_to_tipoff'), c.get('edge'))


def record_candidates(candidates: List[dict], db) -> int:
    """Upsert a scan's calibrated candidates. Returns the number of rows written."""
    if not CANDIDATE_LOG_ENABLED:
        return 0
    rows = [r for r in (_row(c) for c in candidates) if r is not None]
    if rows:
        with db.get_conn() as conn:
            conn.executemany(_UPSERT, rows)
        logger.info(f"Candidate log: {len(rows)} prop side(s) recorded.")
    return len(rows)


def _logloss(p: float, y: int) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -math.log(p if y else 1 - p)


def model_eval_report(db, since: Optional[str] = None, edge_min: float = 0.05) -> Dict[str, dict]:
    """
    Raw model vs market on graded candidates (excl. pushes/voids): Brier and
    log-loss of the raw model prob against the devigged DK/FD prob for the
    same side (lower is better), and flat ROI of the sides where the raw
    model beats break-even (1/odds) by edge_min.
    """
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT market, side, odds, implied_prob, raw_model_prob, won FROM scan_candidates "
            "WHERE settled_at IS NOT NULL AND NOT push AND raw_model_prob IS NOT NULL "
            "AND implied_prob > 0 AND game_date >= ?",
            (since or '0000-00-00',),
        ).fetchall()
    groups: Dict[str, list] = {'ALL': list(rows)}
    for r in rows:
        groups.setdefault(r['market'], []).append(r)
    out = {}
    print(f"Raw model vs market (graded candidates){' since ' + since if since else ''}:")
    print(f"  {'group':32s} {'n':>6s}  {'brier mdl/mkt':>15s}  {'logloss mdl/mkt':>17s}  "
          f"{'model picks (edge>=' + format(edge_min, '.0%') + ')':>26s}")
    for name, rs in groups.items():
        n = len(rs)
        if not n:
            print("  no graded candidates yet")
            continue
        y = [1 if r['won'] else 0 for r in rs]
        pm = [float(r['raw_model_prob']) for r in rs]
        pk = [float(r['implied_prob']) for r in rs]
        brier_m = sum((p - t) ** 2 for p, t in zip(pm, y)) / n
        brier_k = sum((p - t) ** 2 for p, t in zip(pk, y)) / n
        ll_m = sum(_logloss(p, t) for p, t in zip(pm, y)) / n
        ll_k = sum(_logloss(p, t) for p, t in zip(pk, y)) / n
        picks = [(float(r['odds']), t) for r, p, t in zip(rs, pm, y) if p - 1.0 / float(r['odds']) >= edge_min]
        units = sum((o - 1.0) if t else -1.0 for o, t in picks)
        out[name] = {'n': n, 'brier_model': brier_m, 'brier_market': brier_k,
                     'logloss_model': ll_m, 'logloss_market': ll_k,
                     'picks': len(picks), 'pick_roi': units / len(picks) if picks else 0.0}
        s = out[name]
        print(f"  {name:32s} {n:6d}  {brier_m:.4f}/{brier_k:.4f}  {ll_m:.4f}/{ll_k:.4f}    "
              f"{len(picks):5d} bets  ROI {s['pick_roi']:+.1%}")
    print("  The model only has an edge where its brier/logloss beats the market's.")
    return out
