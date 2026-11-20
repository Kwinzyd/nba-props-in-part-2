"""Chronological training and prospective, price-aware artifact promotion."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import joblib
import numpy as np
from dateutil import tz


def fingerprint(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as fh:
        temporary = fh.name
        json.dump(value, fh, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def chronological_split(metadata, min_dates=10):
    """Group all players/markets in a game and all games on a date together."""
    if not metadata:
        raise ValueError('dated event metadata is required')
    dates = []
    event_dates = {}
    for row in metadata:
        day = datetime.strptime(row['game_date'], '%Y-%m-%d').date().isoformat()
        event = row['event_id']
        if not event or event_dates.setdefault(event, day) != day:
            raise ValueError('event identity is missing or spans multiple dates')
        dates.append(day)
    unique = sorted(set(dates))
    if len(unique) < min_dates:
        raise ValueError('insufficient independent dates')
    first, second = max(1, int(len(unique) * .6)), max(2, int(len(unique) * .8))
    groups = [np.array([i for i, day in enumerate(dates) if day in days], dtype=int)
              for days in (set(unique[:first]), set(unique[first:second]), set(unique[second:]))]
    if any(len(group) == 0 for group in groups):
        raise ValueError('empty chronological partition')
    return groups


def stage_artifact(active_path, artifact, metadata, kind='joblib'):
    """Immutable candidate directory. Training never writes the active path."""
    active_path = Path(active_path).resolve()
    directory = active_path.parent / 'candidates' / active_path.stem / uuid.uuid4().hex
    directory.mkdir(parents=True)
    path = directory / active_path.name
    if kind == 'json':
        atomic_json(path, artifact)
    else:
        joblib.dump(artifact, path)
    manifest = {**metadata, 'artifact_id': fingerprint(path), 'artifact_path': str(path),
                'active_path': str(active_path), 'incumbent_id': fingerprint(active_path),
                'created_at': datetime.now(timezone.utc).isoformat(), 'kind': kind,
                'status': 'awaiting_prospective_evaluation'}
    atomic_json(directory / 'manifest.json', manifest)
    return directory / 'manifest.json'


def train_rate_candidate(owner, X, y, metadata, weights=None, ensemble=False):
    """Forward stacking: bases on earliest dates, meta on later dates, final test untouched."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    from src.models.ml_model import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

    if metadata is None or len(metadata) != len(X) or len(y) != len(X):
        raise ValueError('training rows require aligned dates and event IDs')
    train, meta_rows, test = chronological_split(metadata)
    if len(train) < 50 or len(meta_rows) < 10 or len(test) < 20:
        raise ValueError('insufficient train/meta/test samples')
    matrix, labels = np.asarray(X, float), np.asarray(y, float)
    if matrix.shape != (len(labels), len(FEATURE_NAMES)) or not np.isfinite(matrix).all() or not np.isfinite(labels).all():
        raise ValueError('invalid training features/labels')
    sample_weights = None if weights is None else np.asarray(weights, float)
    if sample_weights is not None and (sample_weights.shape != labels.shape or not np.isfinite(sample_weights).all() or (sample_weights <= 0).any()):
        raise ValueError('invalid sample weights')
    model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=.05,
                            subsample=.8, colsample_bytree=.8, random_state=42, verbosity=0,
                            n_jobs=int(os.getenv('TRAIN_N_JOBS', '2')))
    bases = [('xgb', model)]
    if ensemble:
        bases += [('rf', RandomForestRegressor(n_estimators=150, max_depth=8,
                   min_samples_leaf=5, max_features=.7, random_state=42,
                   n_jobs=int(os.getenv('TRAIN_N_JOBS', '2')))),
                  ('lr', LinearRegression())]
    fit_rows = train if ensemble else np.concatenate([train, meta_rows])
    for _, learner in bases:
        learner.fit(matrix[fit_rows], labels[fit_rows],
                    sample_weight=None if sample_weights is None else sample_weights[fit_rows])
    if ensemble:
        meta = Ridge(alpha=.5)
        meta.fit(np.column_stack([m.predict(matrix[meta_rows]) for _, m in bases]), labels[meta_rows],
                 sample_weight=None if sample_weights is None else sample_weights[meta_rows])
        predictions = meta.predict(np.column_stack([m.predict(matrix[test]) for _, m in bases]))
    else:
        predictions = model.predict(matrix[test])
    baseline = np.array([metadata[i]['baseline_rate'] for i in test], float)
    if not np.isfinite(predictions).all() or not np.isfinite(baseline).all():
        raise ValueError('nonfinite evaluation predictions')
    details = {'validation_version': 1, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
               'fit_end': max(metadata[i]['game_date'] for i in np.concatenate([train, meta_rows])),
               'base_fit_end': max(metadata[i]['game_date'] for i in train),
               'test_start': min(metadata[i]['game_date'] for i in test),
               'test_end': max(metadata[i]['game_date'] for i in test),
               'n_train': len(train), 'n_meta': len(meta_rows), 'n_test': len(test),
               'training_data_id': hashlib.sha256(matrix.tobytes() + labels.tobytes()
                   + json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
               'rate_mse': float(np.mean((np.maximum(predictions, 0) - labels[test]) ** 2)),
               'baseline_rate_mse': float(np.mean((baseline - labels[test]) ** 2)),
               'price_evaluation': 'unavailable_in_game_logs'}
    if ensemble:
        artifact = {'bases': bases, 'meta': meta, 'feature_names': FEATURE_NAMES,
                    'feature_schema_version': FEATURE_SCHEMA_VERSION, 'training_metadata': details}
        active = owner.ensemble_path
    else:
        model.prop_feature_schema_version = FEATURE_SCHEMA_VERSION
        model.training_metadata = details
        artifact, active = model, owner.model_path
    return stage_artifact(active, artifact, details)


def _aware(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('timestamps must be timezone-aware')
    return dt.astimezone(timezone.utc)


def promote_candidate(manifest_path, observations, min_samples=30, min_games=10):
    """Consume a new prospective cohort once; fail closed on absent/stale evidence.

    Observations must be stored, pregame paired predictions tagged with artifact
    IDs, not retrospective predictions reconstructed after outcomes were known.
    The shadow collector supplies these records; training does not call this.
    """
    manifest_path = Path(manifest_path).resolve()
    info = json.loads(manifest_path.read_text(encoding='utf-8'))
    active, artifact = Path(info['active_path']), Path(info['artifact_path'])
    lock = active.with_name(active.name + '.promotion.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return {'promoted': False, 'reason': 'promotion_locked'}
    try:
        if not isinstance(observations, list) or any(not isinstance(row, dict) for row in observations):
            raise ValueError('observations must be a JSON array of prediction records')
        if fingerprint(artifact) != info['artifact_id'] or fingerprint(active) != info['incumbent_id']:
            return {'promoted': False, 'reason': 'artifact_changed'}
        ledger_path = active.with_name(active.name + '.evaluation.json')
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        created = _aware(info['created_at'])
        rows, identities, event_dates = [], set(), {}
        for row in observations:
            if row.get('push'):
                continue
            day = datetime.strptime(row['game_date'], '%Y-%m-%d').date().isoformat()
            if day <= max(info['fit_end'], info.get('test_end', info['fit_end']), ledger.get('consumed_through', '0001-01-01')):
                raise ValueError('evaluation dates overlap fitting, prior testing or consumed data')
            predicted, tip, settled = (_aware(row[k]) for k in ('predicted_at', 'commence_time', 'settled_at'))
            if not created <= predicted < tip <= settled:
                raise ValueError('predictions are not prospective/pregame')
            if settled > datetime.now(timezone.utc) or day != tip.astimezone(tz.gettz('America/New_York')).date().isoformat():
                raise ValueError('event date or settlement timestamp is invalid')
            if row['candidate_id'] != info['artifact_id'] or row['incumbent_id'] != info['incumbent_id']:
                raise ValueError('prediction artifact identity mismatch')
            identity = tuple(row[k] for k in ('event_id', 'player_name', 'market', 'line', 'side'))
            if not row['event_id'] or event_dates.setdefault(row['event_id'], day) != day:
                raise ValueError('event identity spans different dates')
            if identity in identities:
                raise ValueError('duplicate evaluation observations')
            identities.add(identity)
            odds, probability = float(row['odds']), float(row['candidate_prob'])
            if not math.isfinite(odds) or odds <= 1:
                raise ValueError('invalid odds')
            incumbent = float(row['incumbent_prob']) if info['incumbent_id'] is not None else 1 / odds
            if not all(math.isfinite(v) for v in (odds, probability, incumbent)) or odds <= 1 or not 0 <= probability <= 1 or not 0 <= incumbent <= 1 or row['won'] not in (0, 1, False, True):
                raise ValueError('invalid evaluation observation')
            rows.append((row['event_id'], day, probability, incumbent, 1 / odds, int(row['won'])))
        if len(rows) < min_samples or len({r[0] for r in rows}) < min_games:
            return {'promoted': False, 'reason': 'insufficient_prospective_prices'}
        # Weight games equally; correlated opposite sides/lines do not become
        # additional independent games merely by increasing the row count.
        scores = {}
        for name, index in (('candidate', 2), ('incumbent', 3), ('market', 4)):
            games = {}
            for row in rows:
                p, y = float(np.clip(row[index], 1e-6, 1 - 1e-6)), row[5]
                games.setdefault(row[0], []).append(((row[index] - y) ** 2, -math.log(p if y else 1 - p)))
            scores[name] = np.mean([np.mean(v, axis=0) for v in games.values()], axis=0).tolist()
        passed = all(scores['candidate'][0] < scores[b][0] - 1e-6 and scores['candidate'][1] <= scores[b][1]
                     for b in ('incumbent', 'market'))
        report = {'promoted': False, 'reason': 'passed' if passed else 'baseline_not_beaten',
                  'scores_brier_logloss': scores, 'candidate_id': info['artifact_id'],
                  'consumed_through': max(r[1] for r in rows), 'n': len(rows)}
        # Consume the cohort on failure too, before publishing, to prevent
        # trying many adjusted candidates against the same test outcomes.
        atomic_json(ledger_path, report)
        if not passed:
            return report
        if active.exists():
            shutil.copy2(active, str(active) + '.bak')
        with tempfile.NamedTemporaryFile(dir=active.parent, delete=False) as fh:
            temporary = fh.name
        try:
            shutil.copyfile(artifact, temporary)
            os.replace(temporary, active)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        report['promoted'] = True
        try:
            atomic_json(manifest_path.parent / 'promotion.json', report)
        except OSError as exc:
            # Publishing already succeeded; do not report that active remained
            # unchanged merely because this additional report could not save.
            report['audit_report_error'] = str(exc)
        return report
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        return {'promoted': False, 'reason': 'invalid_evidence', 'detail': str(exc)}
    finally:
        os.close(fd)
        lock.unlink()
