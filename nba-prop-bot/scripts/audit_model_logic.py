"""Offline reproductions for the November 15 model audit; makes no API calls.

Run from nba-prop-bot: python -m scripts.audit_model_logic
These diagnostics describe current behavior; they do not change model files.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import poisson

from src.clients.rotation_model import RotationModel
from src.models import distributions as dist
from src.models import edge_ranker
from src.models.calibration_model import anchored_prob
from src.models.ml_model import EnsembleModel, FEATURE_NAMES, PropMLModel


def main():
    findings = {}
    anchor = json.loads(Path('src/models/calibration_market.json').read_text())
    findings['calibration'] = {
        'k': anchor['k'], 'b_under': anchor['b_under'],
        'under_raw_060': anchored_prob(0.60, 0.5, True, anchor),
        'under_raw_090': anchored_prob(0.90, 0.5, True, anchor),
    }

    candidate = dict(model_prob=0.53, implied_prob=0.45, odds=1.8,
                     projected_minutes=30, market='player_rebounds', side='UNDER',
                     hours_to_tipoff=2, injury_status='healthy')
    with patch.object(edge_ranker, '_clv_loaded', True), \
         patch.object(edge_ranker, '_clv_edge_mins', {}):
        ranked = edge_ranker.rank_edges([candidate])[0]
    findings['negative_ev_clears_floor'] = {
        k: ranked[k] for k in ('edge', 'ev', 'edge_vs_breakeven', 'edge_min_applied')
    }

    real_rng = np.random.default_rng
    with patch.object(dist.np.random, 'default_rng', lambda *args, **kwargs: real_rng(2026)):
        low_variance = dist.get_probability_distribution(
            'player_points', 20, 25.5, proj_minutes=32,
            player_foul_rate=0.083, variance_scale=1)
        high_variance = dist.get_probability_distribution(
            'player_points', 20, 25.5, proj_minutes=32,
            player_foul_rate=0.083, variance_scale=4)
    findings['mc_variance_scale'] = {'scale_1': low_variance, 'scale_4': high_variance}

    logs = pd.DataFrame({'PTS': [10, 20] * 10, 'MIN': [30] * 20})
    with patch.object(dist.np.random, 'choice', lambda data, **kw: np.tile([10, 20], kw['size'] // 2)):
        findings['bootstrap_ignores_projected_mean'] = {
            'mean_5': dist.get_probability_distribution('player_points', 5, 15.5, logs=logs),
            'mean_30': dist.get_probability_distribution('player_points', 30, 15.5, logs=logs),
        }

    pmf_push = float(poisson.pmf(1, 1))
    findings['integer_line'] = {
        'returned': dist.poisson_over_under(1, 1),
        'true_under_win_unconditional': float(poisson.cdf(0, 1)),
        'push': pmf_push,
        'true_under_win_given_no_push': float(poisson.cdf(0, 1) / (1 - pmf_push)),
    }
    findings['probability_cap_mass'] = dist.get_probability_distribution(
        'player_rebounds', 1, 8.5)

    rotation = RotationModel(None)
    slots = {1: {f's{i}': 0.5 for i in range(24)},
             2: {f's{i}': 1.0 for i in range(24)},
             3: {f's{i}': 1.0 for i in range(24)}}
    findings['rotation_minutes_two_absences'] = rotation._compute_projection(1, [2, 3], slots)
    try:
        rotation.get_projected_minutes(1, [2], logs.assign(TEAM_ABBREVIATION='BOS'), '2026-27', None)
    except KeyError as exc:
        findings['bdl_shaped_logs_rotation_error'] = str(exc)

    # __new__ avoids loading any fitted artifact; feature building needs only col.
    model = PropMLModel.__new__(PropMLModel)
    model.col = 'PTS'
    feature_logs = logs.assign(FGA=15, FTA=4, TOV=2)
    findings['foul_feature_mismatch'] = {
        'training_without_PF': model.build_features(feature_logs)['player_foul_rate'],
        'live_default_without_PF': dist.compute_player_foul_rate(feature_logs),
    }

    # Replace expensive learners and persistence, retaining the actual training
    # control flow. The advertised guard should consult its stored baseline.
    import tempfile
    from sklearn.dummy import DummyRegressor
    factory = lambda **kwargs: DummyRegressor()
    with tempfile.TemporaryDirectory() as tmp, \
         patch('xgboost.XGBRegressor', factory), \
         patch('sklearn.ensemble.RandomForestRegressor', factory), \
         patch('src.data.db.DatabaseClient') as database, \
         patch('joblib.dump') as save:
        ensemble = EnsembleModel.__new__(EnsembleModel)
        ensemble.market = 'player_points'
        ensemble.ensemble_path = str(Path(tmp) / 'candidate.pkl')
        matrix = real_rng(42).normal(size=(60, len(FEATURE_NAMES))).tolist()
        trained = ensemble.train(matrix, np.linspace(0.1, 0.9, 60).tolist())
        findings['ensemble_guard'] = {
            'training_succeeded': trained,
            'baseline_database_calls': database.call_count,
            'model_save_calls': save.call_count,
        }
    print(json.dumps(findings, indent=2))


if __name__ == '__main__':
    main()
