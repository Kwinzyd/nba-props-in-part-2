from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.clients.bdl_game_logs import BDLGameLogs
from src.clients.rotation_model import RotationModel, _slot_key, TOTAL_SLOTS
from src.data.db import DatabaseClient
from src.models import distributions as d
from src.models.ml_model import PropMLModel, EnsembleModel, FEATURE_NAMES, FEATURE_SCHEMA_VERSION


def stat(gid=2, when='2026-11-14', postseason=False, pf=3):
    return dict(min='30:30', pts=20, reb=5, ast=4, fg3m=2, blk=1, stl=1,
                fga=15, fta=3, turnover=2, pf=pf, team={'abbreviation': 'BOS'},
                game={'id': gid, 'date': when, 'season': 2026, 'postseason': postseason,
                      'status': 'Final', 'home_team': {'abbreviation': 'BOS'},
                      'visitor_team': {'abbreviation': 'NYK'}})


def test_fresh_and_persistent_bdl_logs_match(tmp_path):
    db = DatabaseClient(str(tmp_path / 'cache.db'))
    client = MagicMock()
    client.get_game_stats.return_value = [stat(), stat(1, '2026-11-12', True, 0)]
    fresh = BDLGameLogs(client, db).get_player_game_logs(42, 2026)
    cached = BDLGameLogs(client, db).get_player_game_logs(42, 2026)
    pd.testing.assert_frame_equal(fresh, cached)
    assert list(cached['GAME_ID']) == [2, 1]
    assert list(cached['SEASON_ID']) == ['22026', '42026']
    assert list(cached['PF']) == [3, 0]
    assert cached.attrs['game_id_provider'] == 'bdl'
    assert client.get_game_stats.call_count == 1


def test_recent_ids_follow_dates_not_response_order():
    client = MagicMock()
    client.get_game_stats.return_value = [stat(1, '2026-11-12'), stat(2)]
    logs = BDLGameLogs(client)
    assert logs.get_recent_game_ids(42, 2026) == [2, 1]


def test_legacy_permanent_cache_refetches_new_fields():
    db, client = MagicMock(), MagicMock()
    db.get_cached_bdl_game_logs.return_value = [{'game_id': 1, 'min': 30}]
    client.get_game_stats.return_value = [stat()]
    df = BDLGameLogs(client, db).get_player_game_logs(42, 2026, ignore_ttl=True)
    assert 'PF' in df and 'SEASON_ID' in df
    client.get_game_stats.assert_called_once()


def test_missing_fouls_are_not_zero_fouls():
    missing = pd.DataFrame({'PF': [None] * 15, 'MIN': [30] * 15})
    assert d.compute_player_foul_rate(missing) == pytest.approx(0.083)
    missing['PF'] = 0
    assert d.compute_player_foul_rate(missing) == 0


def test_training_and_live_foul_feature_parity():
    model = PropMLModel.__new__(PropMLModel)
    model.col = 'PTS'
    model.market = 'player_points'
    logs = pd.DataFrame({'MIN': [30] * 16, 'PTS': [20] * 16,
                         'PF': [3] * 16, 'GAME_DATE': pd.date_range('2026-11-01', periods=16)[::-1]})
    X, _ = model.build_training_data(logs)
    last_history = logs.iloc[1:]
    live = model.build_features(last_history, player_foul_rate=d.compute_player_foul_rate(last_history))
    assert X[-1][FEATURE_NAMES.index('player_foul_rate')] == live['player_foul_rate'] == pytest.approx(0.1)


def test_incompatible_artifacts_are_not_used(tmp_path, monkeypatch):
    from src.models import ml_model as ml
    monkeypatch.setattr(ml, 'MODEL_DIR', str(tmp_path))
    from sklearn.dummy import DummyRegressor
    old = DummyRegressor().fit([[0], [1]], [0, 1])
    joblib.dump(old, tmp_path / 'player_points_xgb.pkl')
    joblib.dump({'feature_names': FEATURE_NAMES}, tmp_path / 'player_points_ensemble.pkl')
    assert PropMLModel('player_points').model is None
    assert EnsembleModel('player_points').ensemble is None
    old.prop_feature_schema_version = FEATURE_SCHEMA_VERSION
    joblib.dump(old, tmp_path / 'player_points_xgb.pkl')
    assert PropMLModel('player_points').model is not None


def test_bdl_identifiers_never_reach_nba_endpoint():
    logs = pd.DataFrame({'GAME_ID': [123], 'TEAM_ABBREVIATION': ['BOS']})
    logs.attrs['game_id_provider'] = 'bdl'
    model = RotationModel(None)
    model._fetch_pbp = MagicMock()
    assert model.get_projected_minutes(42, [43], logs, '2026-27', MagicMock()) == 0
    model._fetch_pbp.assert_not_called()


def test_rotation_is_joint_bounded_and_conserves_slots():
    slots = {pid: {_slot_key(i): 5 / 8 for i in range(TOTAL_SLOTS)} for pid in range(8)}
    model = RotationModel(None)
    result = model._allocate_minutes([0, 1], slots)
    assert sum(result.values()) == pytest.approx(240)
    assert all(0 <= v <= 48 for v in result.values())
    assert 0 not in result and 1 not in result
    assert len(set(round(v, 6) for v in result.values())) == 1  # ties share vacancies


def test_existing_full_time_starters_do_not_absorb_vacancies():
    slots = {pid: {_slot_key(i): (1 if pid < 4 else 0.5) for i in range(TOTAL_SLOTS)} for pid in range(6)}
    result = RotationModel(None)._allocate_minutes([0], slots)
    assert result[1] == 48 and result[4] == pytest.approx(48) and result[5] == pytest.approx(48)
    assert sum(result.values()) == pytest.approx(240)


@pytest.mark.parametrize('fn', [d.poisson_over_under, lambda m, l: d.negative_binomial_over_under(m, m * 2, l), d.zip_over_under])
def test_integer_push_is_not_under_win(fn):
    result = fn(1, 1)
    assert result['prob_push'] > 0
    if fn is d.poisson_over_under:
        assert result['prob_push'] == pytest.approx(poisson.pmf(1, 1))
        assert result['prob_under'] == pytest.approx(poisson.pmf(0, 1) / (1 - poisson.pmf(1, 1)))


def test_projection_centered_fallback_is_deterministic():
    logs = pd.DataFrame({'PTS': [10, 20] * 10})
    low = d.get_probability_distribution('player_points', 5, 15.5, logs=logs)
    high = d.get_probability_distribution('player_points', 30, 15.5, logs=logs)
    assert low['prob_over'] < high['prob_over']
    assert high == d.get_probability_distribution('player_points', 30, 15.5, logs=logs)


def test_fallback_preserves_adjusted_mean_near_zero():
    logs = pd.DataFrame({'PTS': [0, 40] * 10})
    results = [d.bootstrap_over_under(logs, 'PTS', k, mean=3) for k in range(81)]
    # Tail sum recovers E[X] for a nonnegative integer-valued distribution.
    expectation = sum(r['prob_over'] * (1 - r['prob_push']) for r in results)
    assert expectation == pytest.approx(3)


def test_monte_carlo_variance_is_connected_and_reproducible():
    kwargs = dict(market='player_points', mean=20, line=30.5, proj_minutes=32, player_foul_rate=0.083)
    tight = d.get_probability_distribution(**kwargs, variance_scale=1)
    wide = d.get_probability_distribution(**kwargs, variance_scale=4)
    assert tight == d.get_probability_distribution(**kwargs, variance_scale=1)
    assert wide['prob_over'] > tight['prob_over']


def test_capped_probabilities_remain_complementary():
    r = d.get_probability_distribution('player_rebounds', 1, 8.5)
    assert r['prob_over'] + r['prob_under'] == pytest.approx(1)


def test_zero_mean_is_valid_degenerate_distribution():
    assert d.get_probability_distribution('player_points', 0, 0.5) == {
        'prob_over': 0, 'prob_under': 1, 'prob_push': 0}
    assert d.get_probability_distribution('player_points', 0, 0)['prob_push'] == 1


def test_certain_push_is_not_actionable():
    from src.pipelines.selection_guard import value_rejection
    assert value_rejection({'prob_push': 1}) == 'certain_push'


def test_heuristic_minutes_cannot_exceed_regulation():
    from src.models.projections import estimate_projected_minutes
    logs = pd.DataFrame({'MIN': [47] * 20})
    assert estimate_projected_minutes(logs, logs, 'healthy', starter_flag=True) == 48


def test_shared_simulation_produces_monotonic_lines():
    probs = [d.get_probability_distribution('player_rebounds', 6, line, proj_minutes=30,
                                          player_foul_rate=0.083)['prob_over']
             for line in [3.5, 4.5, 5.5, 6.5, 7.5]]
    assert probs == sorted(probs, reverse=True)
