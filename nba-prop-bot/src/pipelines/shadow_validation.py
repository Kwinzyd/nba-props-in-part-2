"""Immutable, prospective paired model predictions. Never sends or executes bets."""
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
import re
from pathlib import Path

import joblib
import numpy as np
from dateutil import tz

from src import config
from src.models import calibration_model as calibration
from src.models.distributions import get_probability_distribution
from src.models.ml_model import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from src.models.model_validation import fingerprint, promote_candidate
from src.pipelines.selection_guard import quote_rejection, timing_rejection, utc_time
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
ET = tz.gettz('America/New_York')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


@lru_cache(maxsize=200)
def _file_hash(path, size, modified):
    return fingerprint(path)


def _identity(path):
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def serving_context(model_dir=None):
    """Fingerprint serving code, active artifacts and probability configuration."""
    root = Path(model_dir or MODEL_DIR)
    source = Path(__file__).resolve().parents[1]
    files = [*root.glob('*.pkl'), *root.glob('*.json'), *root.glob('xgb/*.pkl')]
    files += list(source.rglob('*.py'))
    details = {'files': {str(p.resolve()): _identity(p) for p in files if not p.name.endswith('.evaluation.json')},
               'feature_schema': FEATURE_SCHEMA_VERSION,
               'configuration': {name: getattr(config, name) for name in
                 ('MIN_PLAYER_GAMES', 'TEAM_PRIOR_GAMES', 'ELIMINATION_MODE',
                  'BDL_PROFILES_ENABLED', 'ALERT_LEAD_HOURS')},
               'raw_edge_gate': calibration.MIN_RAW_EDGE}
    details['playoff_mode'] = os.getenv('PLAYOFF_MODE', 'false').lower()
    return _hash(details), details


def _manifest(path, root):
    info = json.loads(Path(path).read_text(encoding='utf-8'))
    active, artifact = Path(info['active_path']).resolve(), Path(info['artifact_path']).resolve()
    if not active.is_relative_to(root.resolve()) or not artifact.is_relative_to(root.resolve()):
        raise ValueError('candidate path escapes model directory')
    if _identity(artifact) != info['artifact_id'] or _identity(active) != info['incumbent_id']:
        raise ValueError('candidate or incumbent changed')
    if active.name == 'calibration_market.json':
        key = 'anchor'
    elif active.name.endswith(('_ensemble.pkl', '_xgb.pkl')):
        key = active.stem.rsplit('_', 1)[0]
        if info.get('feature_schema_version') != FEATURE_SCHEMA_VERSION:
            raise ValueError('incompatible candidate schema')
        if active.name.endswith('_xgb.pkl'):
            primary = active.with_name(key + '_ensemble.pkl')
            if primary.exists():
                model = _artifact(str(primary), _identity(primary))
                if isinstance(model, dict) and model.get('feature_schema_version') == FEATURE_SCHEMA_VERSION:
                    raise ValueError('standalone candidate is masked by an active ensemble')
    else:
        raise ValueError('inactive or unsupported calibration/model path')
    return {**info, 'manifest_path': str(Path(path).resolve()), 'model_key': key}


def experiments(db, model_dir=None, context=None):
    """Freeze one candidate per market/context until its cohort is evaluated."""
    root = Path(model_dir or MODEL_DIR)
    context_id, details = context or serving_context(root)
    with db.get_conn() as conn:
        conn.execute('INSERT INTO model_shadow_contexts (context_id, manifest) VALUES (?, ?) '
                     'ON CONFLICT (context_id) DO NOTHING', (context_id, _json(details)))
        conn.execute("UPDATE model_shadow_experiments SET status = 'paused_context_changed' "
                     "WHERE status = 'collecting' AND context_id <> ?", (context_id,))
        saved = [dict(r) for r in conn.execute('SELECT * FROM model_shadow_experiments').fetchall()]
    live = {}
    for row in saved:
        if row['status'] != 'collecting' or row['context_id'] != context_id:
            continue
        try:
            info = _manifest(row['manifest_path'], root)
            live[info['model_key']] = {**info, 'experiment_id': row['experiment_id'],
                                      'context_id': context_id, 'context': details}
        except Exception as exc:
            logger.warning('Shadow experiment cannot continue: %s', type(exc).__name__)
            with db.get_conn() as conn:
                conn.execute("UPDATE model_shadow_experiments SET status = 'paused_artifact_changed' WHERE experiment_id = ?", (row['experiment_id'],))
    seen = {r['experiment_id'] for r in saved}
    choices = []
    for path in root.glob('**/candidates/**/manifest.json'):
        try:
            info = _manifest(path, root)
            identifier = _hash([info['manifest_path'], context_id])
            if identifier not in seen:
                choices.append({**info, 'experiment_id': identifier, 'context_id': context_id, 'context': details})
        except Exception as exc:
            logger.debug('Shadow candidate ignored: %s (%s)', path, type(exc).__name__)
    # Prefer an ensemble over its standalone fallback; freeze the latest valid
    # candidate initially, but do not replace a collecting experiment nightly.
    choices.sort(key=lambda x: (x['active_path'].endswith('_ensemble.pkl'), x['created_at']), reverse=True)
    for info in choices:
        if info['model_key'] in live:
            continue
        with db.get_conn() as conn:
            conn.execute('INSERT INTO model_shadow_experiments '
                         '(experiment_id, candidate_id, active_path, manifest_path, context_id) '
                         'VALUES (?, ?, ?, ?, ?) ON CONFLICT (experiment_id) DO NOTHING',
                         (info['experiment_id'], info['artifact_id'], info['active_path'],
                          info['manifest_path'], context_id))
        live[info['model_key']] = info
        logger.info('Shadow experiment collecting: %s [%s]', info['model_key'], info['artifact_id'][:12])
    return list(live.values())


@lru_cache(maxsize=40)
def _artifact(path, identity):
    if fingerprint(path) != identity:
        raise ValueError('candidate bytes changed')
    return json.loads(Path(path).read_text(encoding='utf-8')) if path.endswith('.json') else joblib.load(path)


@lru_cache(maxsize=10000)
def _rate_prediction(path, identity, feature_values):
    artifact = _artifact(path, identity)
    vector = np.array([feature_values], float)
    if isinstance(artifact, dict):
        rates = np.array([[learner.predict(vector)[0] for _, learner in artifact['bases']]])
        return float(artifact['meta'].predict(rates)[0])
    return float(artifact.predict(vector)[0])


def candidate_probability(candidate, experiment):
    """Evaluate the frozen candidate through the same distribution/calibration path."""
    artifact = _artifact(experiment['artifact_path'], experiment['artifact_id'])
    raw = float(candidate['raw_model_prob'])
    details = candidate['_shadow_inputs']
    if experiment['model_key'] == 'anchor':
        implied = 1 / candidate['odds']
        if raw - implied < calibration.MIN_RAW_EDGE:
            return implied, raw, candidate['mean']
        p = calibration.anchored_prob(raw, implied, candidate['side'].upper() == 'UNDER', artifact)
        return float(np.clip(p, .01, .99)), raw, candidate['mean']
    features = details['features']
    rate = _rate_prediction(experiment['artifact_path'], experiment['artifact_id'],
                            tuple(float(features[k]) for k in FEATURE_NAMES))
    if not math.isfinite(rate):
        raise ValueError('nonfinite candidate rate')
    ml_mean = max(0, rate) * details['ml_minutes']
    pre = (((1 - details['ml_weight']) * details['baseline_mean'] + details['ml_weight'] * ml_mean)
           if ml_mean > 0 else details['baseline_mean'])
    mean = pre * candidate['mean'] / details['actual_pre_post_mean']
    probabilities = get_probability_distribution(candidate['market'], mean, candidate['line'],
                                                  logs=details['logs'], **details['distribution'])
    raw = probabilities['prob_over' if candidate['side'].upper() == 'OVER' else 'prob_under']
    scored = calibration.calibrate_edge_candidate({**candidate, 'model_prob': raw},
                                                  playoff_mode=details['playoff_mode'])
    return scored['model_prob'], raw, mean


def record_predictions(candidates, db, refresh_quote, selected=None, model_dir=None):
    if not config.MODEL_SHADOW_ENABLED:
        return 0
    root = Path(model_dir or MODEL_DIR)
    context = serving_context(root)
    selected = experiments(db, root, context) if selected is None else selected
    inserted = 0
    for c in candidates:
        if '_shadow_inputs' not in c or timing_rejection(c):
            continue
        if c['game_date'] != utc_time(c['commence_time']).astimezone(ET).date().isoformat():
            continue
        matching = [e for e in selected if e['model_key'] in ('anchor', c['market']) and e['context_id'] == context[0]]
        if not matching:
            continue
        try:
            quote = refresh_quote(c)
            if quote_rejection(c, quote) or timing_rejection(c):
                continue
        except Exception as exc:
            logger.debug('Shadow quote unavailable: %s', type(exc).__name__)
            continue
        for experiment in matching:
            try:
                with db.get_conn() as conn:
                    existing = conn.execute('SELECT 1 FROM model_shadow_predictions WHERE experiment_id = ? '
                        'AND event_id = ? AND player_name = ? AND market = ? AND line = ? AND side = ?',
                        (experiment['experiment_id'], c['event_id'], c['player_id'], c['market'], c['line'], c['side'])).fetchone()
                if existing:
                    continue
                p, raw, mean = candidate_probability(c, experiment)
                now = datetime.now(timezone.utc)
                if not all(math.isfinite(float(v)) for v in (p, raw, mean, c['model_prob'], c['odds'])):
                    continue
                if now < utc_time(experiment['created_at']) or c['game_date'] <= max(experiment['fit_end'], experiment.get('test_end', '0001-01-01')):
                    continue
                if quote_rejection(c, quote, now) or timing_rejection(c, now):
                    continue
                if serving_context(root)[0] != experiment['context_id']:
                    continue
                inputs = c['_shadow_inputs']
                payload = {key: c[key] for key in ('event_id', 'game_date', 'market', 'line', 'side', 'odds', 'book', 'commence_time')}
                payload.update(player_name=c['player_id'], candidate_id=experiment['artifact_id'],
                    incumbent_id=experiment['incumbent_id'], candidate_prob=float(p), incumbent_prob=float(c['model_prob']),
                    candidate_raw=float(raw), incumbent_raw=float(c['raw_model_prob']), candidate_mean=float(mean),
                    incumbent_mean=float(c['mean']), predicted_at=now.isoformat(), context_id=experiment['context_id'],
                    bdl_game_id=c.get('bdl_game_id'), bdl_player_id=c.get('bdl_player_id'),
                    home_team=c.get('home_team'), away_team=c.get('away_team'),
                    quote=quote, inputs={
                        k: v for k, v in inputs.items() if k != 'logs'})
                # Store only the historical columns used by distribution fallback.
                payload['inputs']['logs'] = json.loads(inputs['logs'].to_json(orient='split'))
                text = _json(payload)
                identity = _hash([experiment['experiment_id'], c['event_id'], c['player_id'], c['market'], c['line'], c['side']])
                deadline = min(utc_time(c['commence_time']), utc_time(quote['checked_at']) + timedelta(seconds=config.MAX_PRICE_AGE_SECONDS))
                if quote.get('updated_at'):
                    deadline = min(deadline, utc_time(quote['updated_at']) + timedelta(seconds=config.MAX_PRICE_AGE_SECONDS))
                clock_guard = 'clock_timestamp() < ?::timestamptz' if hasattr(db, '_dsn') else "julianday('now') < julianday(?)"
                with db.get_conn() as conn:
                    cursor = conn.execute('INSERT INTO model_shadow_predictions '
                        '(prediction_id, experiment_id, game_date, event_id, player_name, market, line, side, snapshot_payload, snapshot_hash) '
                        f'SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ? WHERE {clock_guard} '
                        'ON CONFLICT (experiment_id, event_id, player_name, market, line, side) DO NOTHING',
                        (identity, experiment['experiment_id'], c['game_date'], c['event_id'], c['player_id'],
                         c['market'], c['line'], c['side'], text, hashlib.sha256(text.encode()).hexdigest(), deadline.isoformat()))
                    inserted += max(0, cursor.rowcount)
            except Exception as exc:
                logger.warning('Shadow prediction skipped [%s]: %s', experiment['model_key'], exc)
    if inserted:
        logger.info('Prospective shadow predictions recorded: %d (no alerts/stakes)', inserted)
    return inserted


def pending_dates(db, start='0001-01-01', end=None):
    end = end or (datetime.now(ET).date() - timedelta(days=1)).isoformat()
    with db.get_conn() as conn:
        rows = conn.execute('SELECT DISTINCT p.game_date FROM model_shadow_predictions p '
            'LEFT JOIN model_shadow_results r ON r.prediction_id = p.prediction_id '
            'WHERE r.prediction_id IS NULL AND p.game_date BETWEEN ? AND ? ORDER BY p.game_date', (start, end)).fetchall()
    return [str(r['game_date']) for r in rows]


def settle_predictions(db, target_date, final_games, final_stats, now=None):
    from src.pipelines.settle_results import _grade_prop, _normalize_name, _MARKET_TO_STAT
    from src.clients.bdl_game_logs import _parse_minutes
    now = now or datetime.now(timezone.utc)
    game_ids = {str(g['id']) for g in final_games if 'final' in str(g.get('status', '')).lower()}
    with db.get_conn() as conn:
        rows = conn.execute('SELECT p.* FROM model_shadow_predictions p LEFT JOIN model_shadow_results r '
            'ON r.prediction_id = p.prediction_id WHERE p.game_date = ? AND r.prediction_id IS NULL', (target_date,)).fetchall()
    count = 0
    for row in rows:
        payload = json.loads(row['snapshot_payload'])
        if hashlib.sha256(row['snapshot_payload'].encode()).hexdigest() != row['snapshot_hash']:
            continue
        if utc_time(payload['commence_time']) > now:
            continue
        gid = str(payload.get('bdl_game_id'))
        if gid not in game_ids:
            # Odds-API events have a different ID space. Resolve only an exact
            # matchup among final games rather than trusting name-only stats.
            matches = [g for g in final_games if
                _normalize_name((g.get('home_team') or {}).get('full_name', '')) == _normalize_name(payload.get('home_team') or '') and
                _normalize_name((g.get('visitor_team') or {}).get('full_name', '')) == _normalize_name(payload.get('away_team') or '')]
            if len(matches) != 1:
                continue
            gid = str(matches[0]['id'])
        matches = []
        for stat in final_stats:
            if str((stat.get('game') or {}).get('id')) != gid:
                continue
            player = stat.get('player') or {}
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            if ((payload.get('bdl_player_id') is not None and str(player.get('id')) == str(payload['bdl_player_id']))
                    or (payload.get('bdl_player_id') is None and _normalize_name(name) == _normalize_name(payload['player_name']))):
                matches.append(stat)
        if len(matches) != 1:
            continue  # Missing/ambiguous player stats are not automatically voided.
        stat = matches[0]
        if (stat.get('min') is None or not re.fullmatch(r'\d+(?:\.\d+)?(?::[0-5]\d(?:\.\d+)?)?', str(stat['min']).strip())
                or not math.isfinite(_parse_minutes(stat['min']))):
            continue
        required = ['pts', 'reb', 'ast'] if payload['market'] == 'player_points_rebounds_assists' else [_MARKET_TO_STAT[payload['market']]]
        if _parse_minutes(stat['min']) > 0 and any(stat.get(key) is None for key in required):
            continue
        won, actual = _grade_prop(payload['market'], payload['line'], payload['side'], matches[0])
        if not math.isfinite(actual):
            continue
        with db.get_conn() as conn:
            cursor = conn.execute('INSERT INTO model_shadow_results (prediction_id, actual_result, won, push, settled_at) '
                'VALUES (?, ?, ?, ?, ?) ON CONFLICT (prediction_id) DO NOTHING',
                (row['prediction_id'], actual, bool(won), won is None, now.isoformat()))
            count += max(0, cursor.rowcount)
    return count


def evaluate_experiments(db, model_dir=None, auto_promote=None):
    root = Path(model_dir or MODEL_DIR)
    context_id, _ = serving_context(root)
    auto_promote = config.MODEL_SHADOW_AUTO_PROMOTE if auto_promote is None else auto_promote
    cutoff = (datetime.now(ET).date() - timedelta(days=1)).isoformat()
    with db.get_conn() as conn:
        groups = [dict(r) for r in conn.execute("SELECT * FROM model_shadow_experiments WHERE status = 'collecting'").fetchall()]
    reports = []
    for group in groups:
        if group['context_id'] != context_id or serving_context(root)[0] != context_id:
            with db.get_conn() as conn:
                conn.execute("UPDATE model_shadow_experiments SET status = 'paused_context_changed' WHERE experiment_id = ?", (group['experiment_id'],))
            continue
        info = _manifest(group['manifest_path'], root)
        ledger = Path(info['active_path'] + '.evaluation.json')
        consumed = json.loads(ledger.read_text()).get('consumed_through', '0001-01-01') if ledger.exists() else '0001-01-01'
        with db.get_conn() as conn:
            rows = conn.execute('SELECT p.*, r.won, r.push, r.settled_at FROM model_shadow_predictions p '
                'LEFT JOIN model_shadow_results r ON r.prediction_id = p.prediction_id '
                'WHERE p.experiment_id = ? AND p.game_date > ? AND p.game_date <= ? ORDER BY p.game_date, p.prediction_id',
                (group['experiment_id'], consumed, cutoff)).fetchall()
        # Never select only already-settled bets from an incomplete event day.
        missing = [r['game_date'] for r in rows if r['settled_at'] is None]
        complete = [r for r in rows if not missing or r['game_date'] < min(missing)]
        evidence = []
        for row in complete:
            if hashlib.sha256(row['snapshot_payload'].encode()).hexdigest() != row['snapshot_hash']:
                raise ValueError('shadow snapshot integrity check failed')
            payload = json.loads(row['snapshot_payload'])
            evidence.append({**payload, 'won': bool(row['won']), 'push': bool(row['push']), 'settled_at': row['settled_at']})
        if not auto_promote:
            report = {'promoted': False, 'reason': 'automatic_promotion_disabled', 'n_complete': len(evidence)}
        else:
            report = promote_candidate(group['manifest_path'], evidence)
        usable = [row for row in evidence if not row['push']]
        report['n_complete'] = len(usable)
        report['n_games'] = len({row['event_id'] for row in usable})
        # Raw forecasts remain informative diagnostics even when the active
        # k=0 calibrator suppresses their magnitude in final probabilities.
        if usable:
            report['raw_brier'] = {name: float(np.mean([
                (row[key] - int(row['won'])) ** 2 for row in usable]))
                for name, key in (('candidate', 'candidate_raw'), ('incumbent', 'incumbent_raw'))}
        with db.get_conn() as conn:
            status = 'promoted' if report['promoted'] else ('rejected' if report['reason'] == 'baseline_not_beaten' else 'collecting')
            conn.execute('UPDATE model_shadow_experiments SET status = ?, report = ? WHERE experiment_id = ?',
                         (status, _json(report), group['experiment_id']))
        reports.append({'experiment_id': group['experiment_id'], **report})
        if report['promoted']:
            calibration.reload_calibration_model()
            logger.info('Shadow gate promoted candidate %s', info['artifact_id'][:12])
            # Remaining candidates must collect under the new serving context;
            # independently validated changes are not assumed to combine safely.
            break
    return reports


def stage_cached_candidates(db, model_dir=None, target_since=None):
    """Initial staging from cached box scores only; no provider calls/messages.

    Targets default to this season; earlier cached games remain feature history.
    Cache quality is disclosed rather than silently fabricating fouls/game type.
    """
    from src.clients.bdl_game_logs import BDLGameLogs
    from src.models.ml_model import PropMLModel, EnsembleModel, _MARKET_COL
    from src.models.model_validation import atomic_json, train_rate_candidate
    root = Path(model_dir or MODEL_DIR)
    today = datetime.now(ET).date()
    target_since = target_since or f'{today.year if today.month >= 10 else today.year - 1}-10-01'
    datetime.strptime(target_since, '%Y-%m-%d')
    with db.get_conn() as conn:
        rows = [dict(r) for r in conn.execute('SELECT * FROM bdl_game_log_cache ORDER BY player_id, game_date DESC').fetchall()]
    players = {}
    for row in rows:
        players.setdefault(row['player_id'], []).append(row)
    logs = [BDLGameLogs._sqlite_rows_to_df(rs) for rs in players.values()
            if any(r['game_date'] >= target_since for r in rs)]
    output = []
    for market in _MARKET_COL:
        owner = EnsembleModel.__new__(EnsembleModel)
        owner.market, owner.ensemble_path = market, str(root / 'xgb' / f'{market}_ensemble.pkl')
        builder = PropMLModel.__new__(PropMLModel)
        builder.market, builder.col = market, _MARKET_COL[market]
        X, y, metadata = [], [], []
        for df in logs:
            if len(df) < 15:
                continue
            features, labels, info = builder.build_training_data(df, return_metadata=True, target_since=target_since)
            X.extend(features)
            y.extend(labels)
            metadata.extend(info)
        if not X:
            raise ValueError('no current-season cached training targets')
        path = train_rate_candidate(owner, X, y, metadata, ensemble=True)
        manifest = json.loads(path.read_text())
        manifest['data_source'] = {'type': 'cached_bdl_box_scores', 'target_since': target_since,
            'players_with_targets': len(logs), 'cache_rows': len(rows),
            'missing_pf_rows': sum(r.get('pf') is None for r in rows),
            'unknown_game_type_rows': sum(not r.get('season_id') for r in rows)}
        atomic_json(path, manifest)
        output.append(str(path))
        logger.info('Initial shadow candidate staged: %s (%s dated targets)', market, len(X))
    return output
