"""
Pipeline to dynamically fit an Isotonic Regression on settled bet outcomes.
It calculates 'model_prob' from the (edge, odds) in alerts_sent and maps it
directly to the winning percentage stored in 'bet_results'.

Regression guard: the new model must achieve a lower (better) Brier score than
the incumbent on the most-recent 20% of settled bets (held-out test slice).
If it doesn't improve, the old model is kept and a warning is logged.

train_market_anchored_calibration() is what the scheduler runs: it fits
the market-anchored form in calibration_model (k, b_under) on graded props.
train_isotonic_calibration() is the legacy per-market isotonic fit.

Usage:
    python -m src.pipelines.train_calibration
"""

import os
import shutil
from datetime import datetime

import json

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from src.data.db import DatabaseClient, get_db_client
from src.models.calibration_model import ANCHOR_PATH
from src.models.experiment_tracker import training_run
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'calibration_iso.pkl')

_MIN_TRAIN = 100   # minimum settled bets to attempt a fit
_TEST_FRAC = 0.20  # held-out fraction (chronologically most recent)


def _brier(model, X: np.ndarray, y: np.ndarray) -> float:
    preds = np.clip(model.predict(X), 0.0, 1.0)
    return float(np.mean((preds - y) ** 2))


def train_isotonic_calibration() -> bool:
    db = get_db_client()

    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.edge, a.odds, b.won, a.market, a.raw_model_prob
            FROM alerts_sent a
            JOIN bet_results b ON a.id = b.alert_id
            WHERE b.push = FALSE
              AND a.odds IS NOT NULL
              AND a.odds > 1.0
            ORDER BY COALESCE(b.settled_at, a.timestamp) ASC
            """
        ).fetchall()

    if len(rows) < _MIN_TRAIN:
        logger.warning(f"Insufficient settled bets to train IsotonicRegression (n={len(rows)}). Needs {_MIN_TRAIN}.")
        return False

    # Group data by market
    market_data = {}
    for r in rows:
        mkt = r['market']
        # Use stored raw_model_prob when available (correct value);
        # fall back to legacy reconstruction for historical rows without it.
        stored_raw = r['raw_model_prob']
        if stored_raw is not None:
            model_prob = max(0.01, min(0.99, float(stored_raw)))
        else:
            model_prob = max(0.01, min(0.99, float(r['edge']) + 1.0 / float(r['odds'])))
        market_data.setdefault(mkt, []).append((model_prob, float(r['won'])))
        market_data.setdefault('global', []).append((model_prob, float(r['won'])))

    models = {}
    
    for mkt, data in market_data.items():
        if len(data) < 30 and mkt != 'global':
            continue # Skip small markets, they'll use 'global'
            
        X = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        
        # Chronological split
        split = max(1, int(len(X) * (1.0 - _TEST_FRAC)))
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]
        
        candidate = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        try:
            candidate.fit(X_train, y_train)
            brier = _brier(candidate, X_test, y_test) if len(X_test) > 0 else 0
            models[mkt] = candidate
            logger.info(f"Trained {mkt} calibration: n_train={len(X_train)}, brier={brier:.4f}")
        except Exception as e:
            logger.error(f"Failed to train {mkt} calibration: {e}")

    if 'global' not in models:
        return False

    # Regression guard: only replace if new model improves global Brier on test slice
    if os.path.exists(MODEL_PATH):
        try:
            old = joblib.load(MODEL_PATH)
            # Handle both single-model and market-dict formats
            old_global = old['global'] if isinstance(old, dict) else old
            
            # Re-generate test data for global (the hold-out slice)
            global_data = market_data.get('global', [])
            split = max(1, int(len(global_data) * (1.0 - _TEST_FRAC)))
            X_test_g = np.array([d[0] for d in global_data[split:]])
            y_test_g = np.array([d[1] for d in global_data[split:]])
            
            if len(X_test_g) > 0:
                new_brier = _brier(models['global'], X_test_g, y_test_g)
                old_brier = _brier(old_global, X_test_g, y_test_g)
                if new_brier >= old_brier:
                    logger.warning(f"Regression guard: New Brier {new_brier:.4f} >= Old Brier {old_brier:.4f}. Aborting save.")
                    return False
                logger.info(f"Regression guard passed: {new_brier:.4f} < {old_brier:.4f}")
        except Exception as e:
            logger.debug(f"Regression guard error (likely legacy model format): {e}")

    # Back up and save the full models dict
    if os.path.exists(MODEL_PATH):
        shutil.copy2(MODEL_PATH, MODEL_PATH + '.bak')

    joblib.dump(models, MODEL_PATH)
    logger.info(f"Market-aware isotonic calibration updated. Markets: {list(models.keys())}")

    run_name = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with training_run(run_name, tags={"type": "isotonic_calibration"}) as run:
        run.log_params({
            "min_train": _MIN_TRAIN,
            "test_frac": _TEST_FRAC,
            "n_samples": len(rows),
            "markets_trained": ",".join(models.keys()),
        })
        for mkt, candidate in models.items():
            mkt_data = market_data.get(mkt, [])
            if len(mkt_data) > 0:
                split = max(1, int(len(mkt_data) * (1.0 - _TEST_FRAC)))
                X_test = np.array([d[0] for d in mkt_data[split:]])
                y_test = np.array([d[1] for d in mkt_data[split:]])
                if len(X_test) > 0:
                    run.log_metric(f"brier_{mkt}", _brier(candidate, X_test, y_test))
        run.log_artifact(MODEL_PATH)

    return True


def _anchored_pred(theta, imp, raw, under):
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    z = lg(imp) + theta[0] * (lg(raw) - lg(imp)) + theta[1] * under
    return 1.0 / (1.0 + np.exp(-z))


def _fit_anchored(imp, raw, under, y) -> np.ndarray:
    def nll(theta):
        q = np.clip(_anchored_pred(theta, imp, raw, under), 1e-4, 1 - 1e-4)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    # k bounded to [0, 1]: a negative k would invert the model's view.
    return minimize(nll, x0=[0.2, 0.0], method='L-BFGS-B',
                    bounds=[(0.0, 1.0), (-2.0, 2.0)]).x


def train_market_anchored_calibration() -> bool:
    """
    Fit logit(p) = logit(1/odds) + k*(logit(raw) - logit(1/odds)) + b*UNDER
    on graded player props. Saved only if it beats the incumbent parameters
    on the most recent _TEST_FRAC of bets (chronological hold-out).
    """
    db = get_db_client()
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT a.odds, a.side, a.raw_model_prob, b.won
            FROM alerts_sent a
            JOIN bet_results b ON a.id = b.alert_id
            WHERE b.push = FALSE AND a.odds > 1.0
              AND a.raw_model_prob IS NOT NULL
              AND a.market LIKE 'player_%%'
            ORDER BY COALESCE(b.settled_at, a.timestamp) ASC, a.id ASC
            """
        ).fetchall()

    if len(rows) < _MIN_TRAIN:
        logger.warning(f"Market-anchored calibration: only {len(rows)} graded props (need {_MIN_TRAIN}).")
        return False

    imp   = np.array([1.0 / float(r['odds']) for r in rows])
    raw   = np.array([float(r['raw_model_prob']) for r in rows])
    under = np.array([1.0 if str(r['side']).upper().startswith('UNDER') else 0.0 for r in rows])
    y     = np.array([1.0 if r['won'] else 0.0 for r in rows])

    split = max(1, int(len(y) * (1.0 - _TEST_FRAC)))
    held = slice(split, None)
    theta_tr = _fit_anchored(imp[:split], raw[:split], under[:split], y[:split])
    brier = lambda q: float(np.mean((q - y[held]) ** 2))
    new_brier    = brier(_anchored_pred(theta_tr, imp[held], raw[held], under[held]))
    market_brier = brier(imp[held])

    if os.path.exists(ANCHOR_PATH):
        try:
            with open(ANCHOR_PATH, encoding='utf-8') as fh:
                old = json.load(fh)
            old_brier = brier(_anchored_pred([old['k'], old['b_under']], imp[held], raw[held], under[held]))
            if new_brier > old_brier:
                logger.warning(
                    f"Market-anchored regression guard: new Brier {new_brier:.4f} > "
                    f"incumbent {old_brier:.4f}. Keeping incumbent.")
                return False
        except Exception as e:
            logger.debug(f"Market-anchored guard skipped: {e}")

    # Guard passed on the hold-out; refit on everything for the saved params.
    theta = _fit_anchored(imp, raw, under, y)
    params = {
        'type': 'market_anchored',
        'k': round(float(theta[0]), 4),
        'b_under': round(float(theta[1]), 4),
        'n': int(len(y)),
        'holdout_brier': round(new_brier, 4),
        'holdout_market_brier': round(market_brier, 4),
        'fitted_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(ANCHOR_PATH, 'w', encoding='utf-8') as fh:
        json.dump(params, fh, indent=2)
    logger.info(
        f"Market-anchored calibration saved: k={params['k']} b_under={params['b_under']} "
        f"n={params['n']} hold-out Brier {new_brier:.4f} (market alone {market_brier:.4f})")
    return True


if __name__ == "__main__":
    train_market_anchored_calibration()
