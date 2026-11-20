from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.data.db import DatabaseClient
from src.models.model_validation import stage_artifact
from src.models.ml_model import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.pipelines import shadow_validation as sv


def test_postgres_schema_split_preserves_procedural_bodies():
    from src.data.sql_utils import split_schema_sql as _split_schema_sql
    sql = "SELECT 'literal;--not a comment'; -- real comment;\nDO $body$ BEGIN PERFORM 1; PERFORM 2; END $body$;"
    statements = _split_schema_sql(sql)
    assert len(statements) == 2 and 'PERFORM 1; PERFORM 2;' in statements[1]
    assert 'literal;--not a comment' in statements[0]
    with pytest.raises(ValueError):
        _split_schema_sql('DO $$ BEGIN;')


def test_legacy_prop_digest_is_never_resent():
    from src.pipelines.flush_alerts import flush_pending_alerts
    db, bot = MagicMock(), MagicMock()
    db.get_pending_alerts.return_value = [dict(id=1, alert_type='prop', title='old pick',
                                              body='old price', game_date='2026-11-14')]
    assert flush_pending_alerts(db, bot) == 0
    bot.broadcast.assert_not_called()
    db.mark_pending_alerts_sent.assert_not_called()


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = DatabaseClient(str(tmp_path / 'test.db'))
    root = tmp_path / 'models'
    root.mkdir()
    active = root / 'calibration_market.json'
    active.write_text(json.dumps({'k': 0, 'b_under': .1}))
    tip = datetime.now(timezone.utc) + timedelta(hours=1)
    day = tip.astimezone(sv.ET).date()
    path = stage_artifact(active, {'k': .5, 'b_under': 0},
                           {'fit_end': (day - timedelta(days=3)).isoformat(),
                            'test_end': (day - timedelta(days=2)).isoformat()}, kind='json')
    c = dict(player_id='Test Player', event_id='odds-g1', bdl_game_id=123, bdl_player_id=42,
             game_date=day.isoformat(), market='player_points', line=15.5, side='UNDER',
             odds=2.0, book='draftkings', commence_time=tip.isoformat(),
             model_prob=.55, raw_model_prob=.7, mean=14,
             home_team='Home Team', away_team='Away Team', _shadow_inputs={
                 'features': {k: 0 for k in FEATURE_NAMES}, 'baseline_mean': 20,
                 'ml_minutes': 30, 'ml_weight': .5, 'actual_pre_post_mean': 20,
                 'distribution': {}, 'playoff_mode': False,
                 'logs': pd.DataFrame({'PTS': [10, 20] * 10})})
    monkeypatch.setattr(sv.config, 'MODEL_SHADOW_ENABLED', True)
    refresh = MagicMock(side_effect=lambda candidate: {
        'price': candidate['odds'], 'book': candidate['book'],
        'checked_at': datetime.now(timezone.utc).isoformat()})
    return db, root, path, c, refresh


def record(env):
    db, root, path, c, refresh = env
    return sv.record_predictions([c], db, refresh, model_dir=root)


def test_first_pregame_prediction_is_immutable_and_never_an_alert(env):
    db, root, path, c, refresh = env
    assert record(env) == 1
    c['odds'], c['book'], c['model_prob'] = 1.8, 'fanduel', .6
    assert record(env) == 0
    with db.get_conn() as conn:
        row = conn.execute('SELECT * FROM model_shadow_predictions').fetchone()
        payload = json.loads(row['snapshot_payload'])
        assert payload['odds'] == 2.0 and payload['book'] == 'draftkings'
        assert payload['incumbent_prob'] == .55
        assert hashlib.sha256(row['snapshot_payload'].encode()).hexdigest() == row['snapshot_hash']
        assert conn.execute('SELECT COUNT(*) FROM alerts_sent').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM placed_bets').fetchone()[0] == 0
    with pytest.raises(sqlite3.IntegrityError):
        with db.get_conn() as conn:
            conn.execute("UPDATE model_shadow_predictions SET snapshot_payload = '{}' ")
    with pytest.raises(sqlite3.IntegrityError):
        with db.get_conn() as conn:
            conn.execute('DELETE FROM model_shadow_predictions')


def test_old_candidate_not_replaced_mid_cohort(env):
    db, root, path, c, refresh = env
    assert record(env) == 1
    active = root / 'calibration_market.json'
    new = stage_artifact(active, {'k': .8, 'b_under': .1},
                          {'fit_end': '2020-01-01', 'test_end': '2020-01-02'}, kind='json')
    selected = sv.experiments(db, root)
    assert len(selected) == 1 and selected[0]['manifest_path'] == str(path)


@pytest.mark.parametrize('failure', ['started', 'changed', 'stale', 'missing'])
def test_invalid_price_or_time_never_records(env, failure):
    db, root, path, c, refresh = env
    if failure == 'started':
        c['commence_time'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    elif failure == 'changed':
        refresh.side_effect = lambda _: {'price': 1.8, 'book': c['book'], 'checked_at': datetime.now(timezone.utc).isoformat()}
    elif failure == 'stale':
        refresh.side_effect = lambda _: {'price': c['odds'], 'book': c['book'], 'checked_at': '2000-01-01T00:00:00Z'}
    else:
        refresh.side_effect = lambda _: None
    assert record(env) == 0


def test_candidate_must_exist_before_recording(env, monkeypatch):
    db, root, path, c, refresh = env
    info = json.loads(path.read_text())
    info['created_at'] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(info))
    assert record(env) == 0


def final_data(c, game_id=123, player_id=42, minutes='30:00'):
    games = [{'id': game_id, 'status': 'Final', 'home_team': {'full_name': c['home_team']},
              'visitor_team': {'full_name': c['away_team']}}]
    stats = [{'game': {'id': game_id}, 'player': {'id': player_id, 'first_name': 'Test', 'last_name': 'Player'},
              'min': minutes, 'pts': 14}]
    return games, stats


def test_settlement_is_append_only_game_and_player_matched(env):
    db, root, path, c, refresh = env
    assert record(env) == 1
    games, wrong = final_data(c, player_id=999)
    assert sv.settle_predictions(db, c['game_date'], games, wrong, now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3)) == 0
    games, stats = final_data(c)
    assert sv.settle_predictions(db, c['game_date'], games, stats, now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3)) == 1
    assert sv.settle_predictions(db, c['game_date'], games, stats, now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3)) == 0
    with db.get_conn() as conn:
        row = conn.execute('SELECT * FROM model_shadow_results').fetchone()
        assert row['won'] and not row['push']
    with pytest.raises(sqlite3.IntegrityError):
        with db.get_conn() as conn:
            conn.execute('UPDATE model_shadow_results SET won = 0')


def test_missing_stats_stay_pending_and_dnp_row_voids(env):
    db, root, path, c, refresh = env
    assert record(env) == 1
    games, stats = final_data(c, minutes='0:00')
    assert sv.settle_predictions(db, c['game_date'], games, [], now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3)) == 0
    assert sv.pending_dates(db, c['game_date'], c['game_date']) == [c['game_date']]
    assert sv.settle_predictions(db, c['game_date'], games, stats, now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3)) == 1
    assert sv.pending_dates(db, c['game_date'], c['game_date']) == []
    with db.get_conn() as conn:
        assert conn.execute('SELECT push FROM model_shadow_results').fetchone()[0]


def test_context_change_pauses_old_experiment(env, monkeypatch):
    db, root, path, c, refresh = env
    assert record(env) == 1
    old = sv.serving_context(root)
    monkeypatch.setattr(sv, 'serving_context', lambda *_: ('new-context', old[1]))
    assert sv.evaluate_experiments(db, root) == []
    with db.get_conn() as conn:
        assert conn.execute('SELECT status FROM model_shadow_experiments').fetchone()[0] == 'paused_context_changed'


def test_incomplete_day_is_not_used_for_promotion(env, monkeypatch):
    db, root, path, c, refresh = env
    assert record(env) == 1
    # Move the evaluation clock beyond this event date without manufacturing
    # retroactive prediction records. The database snapshot was pregame.
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime.now(timezone.utc) + timedelta(days=2)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    monkeypatch.setattr(sv, 'datetime', Later)
    promote = MagicMock(return_value={'promoted': False, 'reason': 'insufficient_prospective_prices'})
    monkeypatch.setattr(sv, 'promote_candidate', promote)
    sv.evaluate_experiments(db, root)
    assert promote.call_args.args[1] == []
    games, stats = final_data(c)
    sv.settle_predictions(db, c['game_date'], games, stats, now=datetime.fromisoformat(c['commence_time']) + timedelta(hours=3))
    sv.evaluate_experiments(db, root)
    evidence = promote.call_args.args[1]
    assert len(evidence) == 1 and evidence[0]['odds'] == 2.0 and evidence[0]['won']


def test_rate_candidate_reuses_actual_blend_and_post_adjustments(env, monkeypatch):
    db, root, path, c, refresh = env
    active = root / 'xgb' / 'player_points_ensemble.pkl'
    rate = DummyRegressor(strategy='constant', constant=.8).fit(np.zeros((2, len(FEATURE_NAMES))), [.8, .8])
    # Standalone rate artifact suffices to test the mean transformation.
    active = root / 'xgb' / 'player_points_xgb.pkl'
    staged = stage_artifact(active, rate, {'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'fit_end': '2020-01-01', 'test_end': '2020-01-02'})
    selected = [e for e in sv.experiments(db, root) if e['model_key'] == 'player_points'][0]
    monkeypatch.setattr(sv.calibration, '_get_anchor', lambda: None)
    monkeypatch.setattr(sv.calibration, '_get_iso_model', lambda: None)
    p, raw, mean = sv.candidate_probability(c, selected)
    assert mean == pytest.approx((.5 * 20 + .5 * .8 * 30) * 14 / 20)
    assert 0 <= p <= 1 and 0 <= raw <= 1
