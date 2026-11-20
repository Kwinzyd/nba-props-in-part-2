"""Train calibration candidates without replacing active calibration.

Dates are grouped by event day; historical test scores are diagnostics only.
Promotion requires a later prospective cohort with actual pregame prices.
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from src.data.db import get_db_client
from src.models.calibration_model import ANCHOR_PATH
from src.models.model_validation import chronological_split, stage_artifact
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'calibration_iso.pkl')
_MIN_TRAIN = 100
_TEST_FRAC = .20


def _training_rows():
    db = get_db_client()
    with db.get_conn() as conn:
        rows = conn.execute("""
            SELECT a.id, a.event_id, a.market, a.odds, a.side, a.raw_model_prob,
                   COALESCE(DATE(a.game_date), DATE(a.timestamp)) AS game_date, b.won
            FROM alerts_sent a JOIN bet_results b ON a.id = b.alert_id
            WHERE b.push = FALSE AND b.won IS NOT NULL AND a.odds > 1
              AND a.stake > 0 AND a.raw_model_prob IS NOT NULL
              AND a.market LIKE 'player_%%'
            ORDER BY COALESCE(DATE(a.game_date), DATE(a.timestamp)), a.id
        """).fetchall()
    return [dict(r) for r in rows]


def _dataset():
    rows = _training_rows()
    valid = [r for r in rows if np.isfinite(float(r['odds']))
             and np.isfinite(float(r['raw_model_prob'])) and 0 < float(r['raw_model_prob']) < 1]
    event_days = {}
    for row in valid:
        if row['event_id']:
            event_days.setdefault(str(row['event_id']), set()).add(str(row['game_date'])[:10])
    ambiguous = {event for event, days in event_days.items() if len(days) > 1}
    excluded = sum(str(row['event_id']) in ambiguous for row in valid)
    valid = [row for row in valid if str(row['event_id']) not in ambiguous]
    if excluded:
        logger.warning('Calibration: excluded %d rows with ambiguous historical event dates', excluded)
    if len(valid) < _MIN_TRAIN:
        raise ValueError('insufficient eligible settled props')
    metadata = [{'game_date': str(r['game_date'])[:10],
                 'event_id': str(r['event_id'] or ('date:' + str(r['game_date'])[:10]))} for r in valid]
    train, validation, test = chronological_split(metadata)
    fit = np.concatenate([train, validation])
    if len(fit) < 50 or len(test) < 20:
        raise ValueError('insufficient calibration holdout')
    details = {'fit_end': max(metadata[i]['game_date'] for i in fit),
               'test_start': min(metadata[i]['game_date'] for i in test),
               'test_end': max(metadata[i]['game_date'] for i in test),
               'n_fit': len(fit), 'n_test': len(test), 'validation_version': 1}
    details['ambiguous_event_rows_excluded'] = excluded
    raw = np.array([float(r['raw_model_prob']) for r in valid])
    implied = np.array([1 / float(r['odds']) for r in valid])
    under = np.array([str(r['side']).upper() == 'UNDER' for r in valid], float)
    y = np.array([float(r['won']) for r in valid])
    return valid, raw, implied, under, y, fit, test, details


def _anchored_pred(theta, imp, raw, under):
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    z = lg(imp) + theta[0] * (lg(raw) - lg(imp)) + theta[1] * under
    return 1.0 / (1.0 + np.exp(-z))


def _fit_anchored(imp, raw, under, y) -> np.ndarray:
    def nll(theta):
        q = np.clip(_anchored_pred(theta, imp, raw, under), 1e-4, 1 - 1e-4)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    # k bounded to [0, 1]: a negative k would invert the model's view.
    result = minimize(nll, x0=[0.2, 0.0], method='L-BFGS-B',
                      bounds=[(0.0, 1.0), (-2.0, 2.0)])
    if not result.success or not np.isfinite(result.fun):
        raise ValueError('calibration optimization did not converge')
    return result.x


def train_market_anchored_calibration() -> bool:
    """Stage fixed parameters fitted before the final chronological test dates.

    Returns False because active calibration has not been replaced. The
    manifest path is logged for the subsequent prospective shadow evaluation.
    """
    try:
        rows, raw, implied, under, y, fit, test, details = _dataset()
        theta = _fit_anchored(implied[fit], raw[fit], under[fit], y[fit])
        if not np.isfinite(theta).all():
            raise ValueError('nonfinite fitted calibration')
        candidate = _anchored_pred(theta, implied[test], raw[test], under[test])
        details.update(holdout_brier=float(np.mean((candidate - y[test]) ** 2)),
                       holdout_market_brier=float(np.mean((implied[test] - y[test]) ** 2)))
        artifact = {'type': 'market_anchored', 'k': float(theta[0]), 'b_under': float(theta[1]),
                    'n': len(fit), 'fitted_at': datetime.now(timezone.utc).isoformat(),
                    'training_metadata': details, **details}
        path = stage_artifact(ANCHOR_PATH, artifact, details, kind='json')
        logger.info('Calibration candidate staged (active unchanged): %s', path)
    except Exception as exc:
        logger.error('Calibration training failed closed: %s', exc)
    return False


def train_isotonic_calibration() -> bool:
    """Legacy entrypoint now stages a candidate; it cannot bypass promotion."""
    try:
        rows, raw, implied, under, y, fit, test, details = _dataset()
        models = {}
        for market in ['global'] + sorted({r['market'] for r in rows}):
            indices = fit if market == 'global' else np.array([i for i in fit if rows[i]['market'] == market], int)
            if len(indices) < 30:
                continue
            model = IsotonicRegression(y_min=.01, y_max=.99, out_of_bounds='clip')
            model.fit(raw[indices], y[indices])
            models[market] = model
        if 'global' not in models:
            raise ValueError('insufficient global fit')
        predictions = np.array([float(models.get(rows[i]['market'], models['global']).predict([raw[i]])[0]) for i in test])
        details.update(holdout_brier=float(np.mean((predictions - y[test]) ** 2)),
                       holdout_market_brier=float(np.mean((implied[test] - y[test]) ** 2)))
        models['__metadata__'] = details
        path = stage_artifact(MODEL_PATH, models, details)
        logger.info('Isotonic candidate staged (active unchanged): %s', path)
    except Exception as exc:
        logger.error('Isotonic training failed closed: %s', exc)
    return False


if __name__ == '__main__':
    train_market_anchored_calibration()
