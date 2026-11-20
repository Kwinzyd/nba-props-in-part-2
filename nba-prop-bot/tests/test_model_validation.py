from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.models import model_validation as v, ml_model as ml
from src.pipelines import train_calibration as cal


def metadata(n=150):
    return [dict(game_date=(datetime(2026, 1, 1) + timedelta(days=i // 3)).date().isoformat(),
                 event_id=f'g{i // 3}', baseline_rate=.5) for i in range(n)]


def test_split_groups_dates_globally_even_in_player_order():
    rows = metadata()
    rows = rows[::2] + rows[1::2]
    a, b, c = v.chronological_split(rows)
    assert max(rows[i]['game_date'] for i in a) < min(rows[i]['game_date'] for i in b)
    assert max(rows[i]['game_date'] for i in b) < min(rows[i]['game_date'] for i in c)
    assert not {rows[i]['event_id'] for i in a} & {rows[i]['event_id'] for i in b}
    assert set(a) | set(b) | set(c) == set(range(len(rows)))


def test_split_rejects_event_spanning_dates():
    rows = metadata()
    rows[-1]['event_id'] = rows[0]['event_id']
    with pytest.raises(ValueError):
        v.chronological_split(rows)


def test_training_requires_event_dates(tmp_path):
    model = ml.PropMLModel.__new__(ml.PropMLModel)
    model.model_path, model.market = str(tmp_path / 'active.pkl'), 'player_points'
    assert model.train([[1]] * 100, [1] * 100) is False
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('ensemble', [False, True])
def test_training_stages_evaluated_model_without_refit_or_active_write(tmp_path, monkeypatch, ensemble):
    fitted_rows = []
    class RecordingDummy(DummyRegressor):
        def fit(self, X, y, sample_weight=None):
            fitted_rows.append(set(X[:, 0]))
            return super().fit(X, y, sample_weight=sample_weight)
    # Use a serializable standard dummy; record the real fit calls via a wrapper.
    original = DummyRegressor.fit
    def record(self, X, y, sample_weight=None):
        fitted_rows.append(set(X[:, 0]))
        return original(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(DummyRegressor, 'fit', record)
    monkeypatch.setattr('xgboost.XGBRegressor', lambda **kw: DummyRegressor())
    monkeypatch.setattr('sklearn.ensemble.RandomForestRegressor', lambda **kw: DummyRegressor())
    owner = MagicMock()
    active = tmp_path / 'active.pkl'
    active.write_bytes(b'incumbent must remain intact')
    owner.model_path = owner.ensemble_path = str(active)
    matrix = np.zeros((150, len(ml.FEATURE_NAMES)))
    matrix[:, 0] = np.arange(150)
    manifest_path = v.train_rate_candidate(owner, matrix, np.linspace(.1, .9, 150), metadata(), ensemble=ensemble)
    info = json.loads(manifest_path.read_text())
    assert active.read_bytes() == b'incumbent must remain intact'
    assert info['fit_end'] < info['test_start']
    assert all(max(indices) < 120 for indices in fitted_rows)
    if ensemble:
        assert all(max(indices) < 90 for indices in fitted_rows)
    artifact = joblib.load(info['artifact_path'])
    details = artifact['training_metadata'] if ensemble else artifact.training_metadata
    assert details['n_test'] == 30
    assert details['price_evaluation'] == 'unavailable_in_game_logs'
    assert v.fingerprint(info['artifact_path']) == info['artifact_id']


def test_training_features_ignore_target_and_unarchived_context():
    model = ml.PropMLModel.__new__(ml.PropMLModel)
    model.market, model.col = 'player_points', 'PTS'
    logs = pd.DataFrame({'GAME_DATE': pd.date_range('2026-01-01', periods=20)[::-1],
                         'MIN': [30] * 20, 'PTS': list(range(20)), 'PF': [2] * 20,
                         'MATCHUP': ['BOS vs. NYK'] * 20, 'GAME_ID': list(range(20))})
    X, y, info = model.build_training_data(logs, return_metadata=True,
                                          opp_def_lookup={'NYK': 100}, pace_lookup={'NYK': 1000})
    altered = logs.copy()
    altered.loc[0, 'PTS'] = 1000
    X2, y2, _ = model.build_training_data(altered, return_metadata=True,
                                         opp_def_lookup={'NYK': 0}, pace_lookup={'NYK': 0})
    assert X[-1] == X2[-1] and y[-1] != y2[-1]
    assert not set(ml.FEATURE_NAMES) & ml._UNARCHIVED
    assert len(X) == len(info)
    features = model.build_features(logs, real_usage_pct=.99, ts_pct=.99)
    assert features['real_usage_pct'] == model._usage_proxy(logs)
    assert features['ts_pct_5g'] == model._ts_pct(logs)


def staged(tmp_path):
    active = tmp_path / 'active.json'
    active.write_text('{"incumbent": true}')
    manifest = v.stage_artifact(active, {'candidate': True},
                                {'fit_end': '2026-01-01', 'test_end': '2026-01-02'}, kind='json')
    info = json.loads(manifest.read_text())
    info['created_at'] = '2026-01-03T00:00:00+00:00'
    v.atomic_json(manifest, info)
    return manifest, info, active


def observations(info):
    rows = []
    for game in range(10):
        day = datetime(2026, 1, 10, 20, tzinfo=timezone.utc) + timedelta(days=game)
        for player in range(3):
            won = (game + player) % 2
            rows.append(dict(event_id=f'g{game}', player_name=f'p{player}', market='player_points',
                             line=10.5, side='UNDER', game_date=day.date().isoformat(),
                             predicted_at=day.isoformat(), commence_time=(day + timedelta(hours=3)).isoformat(),
                             settled_at=(day + timedelta(hours=6)).isoformat(), odds=2, won=won,
                             candidate_prob=.8 if won else .2, incumbent_prob=.5,
                             candidate_id=info['artifact_id'], incumbent_id=info['incumbent_id']))
    return rows


def test_promotion_needs_real_price_evaluation(tmp_path):
    path, info, active = staged(tmp_path)
    before = active.read_bytes()
    result = v.promote_candidate(path, [])
    assert result['reason'] == 'insufficient_prospective_prices'
    assert active.read_bytes() == before


def test_successful_promotion_is_backed_up_and_identity_checked(tmp_path):
    path, info, active = staged(tmp_path)
    before = active.read_bytes()
    report = v.promote_candidate(path, observations(info))
    assert report['promoted'] is True
    assert active.read_bytes() == Path(info['artifact_path']).read_bytes()
    assert Path(str(active) + '.bak').read_bytes() == before
    assert v.promote_candidate(path, observations(info))['reason'] == 'artifact_changed'


def test_rejection_preserves_incumbent_and_consumes_test_cohort(tmp_path):
    path, info, active = staged(tmp_path)
    rows = observations(info)
    for row in rows:
        row['candidate_prob'] = .4 if row['won'] else .6
    before = active.read_bytes()
    assert v.promote_candidate(path, rows)['reason'] == 'baseline_not_beaten'
    assert active.read_bytes() == before
    assert v.promote_candidate(path, observations(info))['reason'] == 'invalid_evidence'


def test_fresh_later_cohort_can_be_evaluated_after_rejection(tmp_path):
    path, info, _ = staged(tmp_path)
    bad = observations(info)
    for row in bad:
        row['candidate_prob'] = .5
    assert not v.promote_candidate(path, bad)['promoted']
    later = observations(info)
    for row in later:
        for key in ('predicted_at', 'commence_time', 'settled_at'):
            stamp = datetime.fromisoformat(row[key]) + timedelta(days=30)
            row[key] = stamp.isoformat()
        row['game_date'] = (datetime.fromisoformat(row['game_date']) + timedelta(days=30)).date().isoformat()
    assert v.promote_candidate(path, later)['promoted']


def test_candidate_must_beat_incumbent_as_well_as_market(tmp_path):
    path, info, active = staged(tmp_path)
    rows = observations(info)
    for row in rows:
        row['incumbent_prob'] = .95 if row['won'] else .05
    assert v.promote_candidate(path, rows)['reason'] == 'baseline_not_beaten'


@pytest.mark.parametrize('changes', [
    {'candidate_id': 'other'}, {'incumbent_id': 'other'}, {'odds': 0},
    {'candidate_prob': float('nan')}, {'won': None},
    {'predicted_at': '2026-01-01T01:00:00Z'},
    {'predicted_at': '2026-01-10T23:00:00Z'},
    {'game_date': '2026-01-01'}, {'game_date': '2026-01-11'},
    {'settled_at': '2026-01-10T19:00:00Z'},
])
def test_invalid_evidence_fails_closed(tmp_path, changes):
    path, info, active = staged(tmp_path)
    rows = observations(info)
    rows[0].update(changes)
    before = active.read_bytes()
    assert v.promote_candidate(path, rows)['promoted'] is False
    assert active.read_bytes() == before


def test_duplicate_rows_cannot_inflate_sample_count(tmp_path):
    path, info, _ = staged(tmp_path)
    rows = observations(info)
    rows[-1] = rows[0]
    assert v.promote_candidate(path, rows)['reason'] == 'invalid_evidence'


def test_artifact_tamper_blocks_promotion(tmp_path):
    path, info, active = staged(tmp_path)
    Path(info['artifact_path']).write_text('changed')
    assert v.promote_candidate(path, observations(info))['reason'] == 'artifact_changed'


def test_persistence_failure_does_not_publish(tmp_path, monkeypatch):
    path, info, active = staged(tmp_path)
    before = active.read_bytes()
    monkeypatch.setattr(v, 'atomic_json', MagicMock(side_effect=OSError('disk full')))
    assert v.promote_candidate(path, observations(info))['promoted'] is False
    assert active.read_bytes() == before


@pytest.mark.parametrize('isotonic', [False, True])
def test_calibration_stages_without_reusing_holdout_or_overwriting_active(tmp_path, monkeypatch, isotonic):
    rows = []
    for i, meta in enumerate(metadata()):
        rows.append(dict(id=i, event_id=meta['event_id'], game_date=meta['game_date'],
                         market='player_points', odds=2, side='UNDER', raw_model_prob=.7 if i % 2 else .3, won=i % 2))
    monkeypatch.setattr(cal, '_training_rows', lambda: rows)
    active = tmp_path / ('active.pkl' if isotonic else 'active.json')
    active.write_bytes(b'incumbent')
    monkeypatch.setattr(cal, 'MODEL_PATH' if isotonic else 'ANCHOR_PATH', str(active))
    fn = cal.train_isotonic_calibration if isotonic else cal.train_market_anchored_calibration
    assert fn() is False  # false means no active replacement, not a failed stage
    assert active.read_bytes() == b'incumbent'
    manifests = list(tmp_path.glob('candidates/**/manifest.json'))
    assert len(manifests) == 1
    info = json.loads(manifests[0].read_text())
    assert info['n_fit'] + info['n_test'] == len(rows)
    assert info['fit_end'] < info['test_start']
