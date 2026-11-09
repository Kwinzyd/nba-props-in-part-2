from typing import List, Dict, Any
from src.config import (
    MIN_PROJECTED_MINUTES,
    PLAYOFF_MIN_PROJECTED_MINUTES,
    PLAYOFF_EDGE_MIN,
    PER_MARKET_EDGE_MIN,
    ALT_TO_BASE_MARKET,
    ELIMINATION_MODE,
    ELIMINATION_EDGE_BUMP,
)

# Priority 7: db is optional — set once from scan_props
_state: Dict[str, Any] = {'db': None, 'playoff_mode': False}

# CLV-calibrated per-market edge minimums (loaded lazily from settled alerts)
_clv_edge_mins: Dict[str, float] = {}
_clv_loaded: bool = False

# Time-based edge thresholds
# Early-morning lines carry more pre-game uncertainty (injury news not yet priced in).
# Close to tip-off the line is efficient but late info (scratches, warm-up reports)
# can create real, low-latency edges that justify a looser threshold.
_EARLY_EDGE_MIN  = 0.05   # > 4 hours to tip
_LATE_EDGE_MIN   = 0.04   # < 1 hour to tip (2% was below calibration uncertainty)
_EARLY_HOURS     = 4.0
_LATE_HOURS      = 1.0


def _load_clv_thresholds() -> None:
    """
    Compute per-market edge floors from 30-day rolling CLV **and** ROI.

    Formula:
      base        = _EARLY_EDGE_MIN (0.05)
      clv_delta   = clamp(-avg_clv * 3, -0.02, +0.03)
      roi_delta   = clamp(-roi * 2,    -0.02, +0.02)
                    — both negative CLV and negative ROI raise the floor
      floor       = clamp(base + clv_delta + roi_delta, 0.02, 0.15)

    The result is stored in _clv_edge_mins[market].  The static
    PER_MARKET_EDGE_MIN overrides in config are applied on top in
    compute_dynamic_edge_min() (take max), so manually tuned hard
    floors are never relaxed below their configured minimum.

    Falls back silently when the DB is unavailable or has < 10 CLV
    records per market.
    """
    global _clv_loaded, _clv_edge_mins
    _clv_loaded = True
    db = _state['db']
    if db is None:
        return
    try:
        per_market_clv = db.get_per_market_clv(days_back=30, min_samples=10)
    except Exception:
        per_market_clv = {}
    try:
        per_market_roi = db.get_per_market_roi(days_back=30, min_samples=30)
    except Exception:
        per_market_roi = {}

    from src.utils.logging_utils import get_logger as _log
    log = _log(__name__)

    all_markets = set(per_market_clv) | set(per_market_roi)
    for market in all_markets:
        avg_clv = per_market_clv.get(market, 0.0)
        roi     = per_market_roi.get(market, 0.0)

        clv_delta = max(-0.02, min(0.06, -avg_clv * 3.0))
        roi_delta = max(-0.02, min(0.04, -roi * 2.0))
        floor = max(0.04, min(0.15, _EARLY_EDGE_MIN + clv_delta + roi_delta))

        _clv_edge_mins[market] = floor
        log.info(
            f"CLV+ROI floor [{market.replace('player_', '')}]: "
            f"avg_clv={avg_clv:+.4f}  roi={roi:+.4f}  "
            f"delta={clv_delta+roi_delta:+.3f}  floor={floor:.3f}"
        )


def reload_clv_thresholds() -> None:
    """Force a fresh reload of CLV-adaptive floors. Called nightly by the scheduler."""
    global _clv_loaded, _clv_edge_mins
    _clv_loaded = False
    _clv_edge_mins = {}
    _load_clv_thresholds()


def compute_dynamic_edge_min(hours_to_tipoff: float, market: str = '') -> float:
    """
    Scale the minimum edge requirement based on how close tip-off is,
    with optional per-market CLV calibration.

    Returns:
        0.05  when hours_to_tipoff >= 4  (early — require strong edge)
        0.02  when hours_to_tipoff <= 1  (pre-game — accept thin edge)
        Linear interpolation between 1 and 4 hours.

    If CLV data exists for this market, uses that as the early-edge floor instead.
    """
    global _clv_loaded
    if not _clv_loaded:
        _load_clv_thresholds()

    early = _clv_edge_mins.get(market, _EARLY_EDGE_MIN) if market else _EARLY_EDGE_MIN
    if _state.get('playoff_mode'):
        # Playoff lines are sharper; require a higher early-edge floor.
        early = max(early, PLAYOFF_EDGE_MIN)
    market_floor = PER_MARKET_EDGE_MIN.get(market, 0.0) if market else 0.0
    if market_floor:
        early = max(early, market_floor)
    late = max(_LATE_EDGE_MIN, market_floor)

    if hours_to_tipoff >= _EARLY_HOURS:
        floor = early
    elif hours_to_tipoff <= _LATE_HOURS:
        floor = late
    else:
        t = (hours_to_tipoff - _LATE_HOURS) / (_EARLY_HOURS - _LATE_HOURS)
        floor = round(late + t * (early - late), 4)

    if ELIMINATION_MODE:
        # Elimination games: sharpest markets of the year + garbage-time risk.
        floor += ELIMINATION_EDGE_BUMP
    return floor


def set_db(db):
    """Call once from scan_props to give edge_ranker access to DB for bias lookups."""
    _state['db'] = db


def set_playoff_mode(playoff_mode: bool) -> None:
    """Call once from scan_props to enable playoff-tightened edge/minutes gates."""
    _state['playoff_mode'] = bool(playoff_mode)


def rank_edges(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter and rank prop candidates.

    edge         = model_prob - devigged implied prob (what the edge floors use)
    edge_vs_be   = model_prob - 1/odds, the edge over the price actually paid;
                   the sort key. Ranking on `edge` let the vig itself count as
                   edge, so high-hold markets (0.5-line steals/blocks) floated
                   to the top, and the old ev/(mean*1.25) key favoured
                   low-count props outright.

    A candidate is dropped when the devigged sharp consensus says its side is
    below break-even at our price — the sharp market disagrees, so there is no
    bet. The old multiplicative boosters (steam, stale line, dispersion, sharp
    book role, consensus x1.15, fragile, per-book bias) were removed on
    2026-11-09: none was validated, and with k=0 calibration they only
    reshuffled which market-priced unders got sent.
    """
    ranked = []

    for c in candidates:
        model_prob   = c.get('model_prob',  0)
        implied_prob = c.get('implied_prob', 0)
        odds         = c.get('odds',        0)
        proj_mins    = c.get('projected_minutes', 0)
        status       = (c.get('injury_status') or 'healthy').lower()
        raw_market   = c.get('market', '')
        market       = ALT_TO_BASE_MARKET.get(raw_market, raw_market)
        side         = str(c.get('side', '')).upper()

        _min_minutes = PLAYOFF_MIN_PROJECTED_MINUTES if _state.get('playoff_mode') else MIN_PROJECTED_MINUTES
        if proj_mins < _min_minutes or "out" in status:
            continue

        breakeven = 1.0 / odds if odds > 1.0 else 1.0

        # Sharp veto: consensus_prob is the devigged sharp P(over) at this line.
        consensus_prob = c.get('consensus_prob')
        if consensus_prob is not None:
            sharp_fair = consensus_prob if side == 'OVER' else 1.0 - consensus_prob
            c['sharp_fair_prob'] = sharp_fair
            if sharp_fair < breakeven:
                continue

        edge = model_prob - implied_prob
        edge_vs_be = model_prob - breakeven
        ev = model_prob * odds - 1.0

        # Injury penalties (risk of a mid-game exit or minutes cap)
        if "questionable" in status or "gtd" in status:
            edge *= 0.8
            edge_vs_be *= 0.8
            ev *= 0.8
        elif "doubtful" in status:
            edge *= 0.5
            edge_vs_be *= 0.5
            ev *= 0.5

        c['edge']               = edge
        c['edge_vs_breakeven']  = edge_vs_be
        c['ev']                 = ev
        c['edge_min_applied']   = compute_dynamic_edge_min(c.get('hours_to_tipoff', 4.0), market=market)

        ranked.append(c)

    ranked.sort(key=lambda x: x.get('edge_vs_breakeven', 0), reverse=True)
    return ranked
