from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from src.data.db import DatabaseClient
from src.pipelines import send_alerts as sa, scan_props as sp
from src.pipelines.selection_guard import quote_rejection, timing_rejection, value_rejection


def candidate(**changes):
    c = dict(player_id='Test Player', event_id='g1', game_date='2026-11-15',
             market='player_rebounds', side='UNDER', line=5.5, book='draftkings',
             model_prob=0.65, implied_prob=0.5, raw_model_prob=0.7, odds=1.9,
             projected_minutes=30, edge=0.12, ev=0.99,
             commence_time=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    c.update(changes)
    return c


def quote(c, **changes):
    q = dict(price=c['odds'], book=c['book'], checked_at=datetime.now(timezone.utc).isoformat())
    q.update(changes)
    return q


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = DatabaseClient(str(tmp_path / 'test.db'))
    monkeypatch.setattr(sa, 'record_execution', MagicMock())
    monkeypatch.setattr(sa, 'BANKROLL', 1000)
    monkeypatch.setattr(sa, 'STAKE_MODE', 'flat')
    monkeypatch.setattr(sa, 'FLAT_STAKE_PCT', 0.01)
    return db, MagicMock()


def assert_rejected(db, bot, reason):
    with db.get_conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM alerts_sent').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM clv_tracking').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM pending_alerts').fetchone()[0] == 0
        assert conn.execute('SELECT reason FROM selection_rejections ORDER BY recorded_at DESC').fetchone()[0] == reason
    bot.broadcast.assert_not_called()
    sa.record_execution.assert_not_called()


@pytest.mark.parametrize('changes,reason', [
    ({'model_prob': 0.5, 'odds': 2}, 'nonpositive_ev'),
    ({'model_prob': 0.53, 'odds': 1.8, 'implied_prob': 0.45}, 'nonpositive_ev'),
    ({'model_prob': float('nan')}, 'invalid_numbers'),
    ({'odds': float('inf')}, 'invalid_numbers'),
    ({'model_prob': 1.2}, 'invalid_numbers'),
    ({'line': float('nan')}, 'invalid_numbers'),
    ({'commence_time': None}, 'unknown_tipoff'),
    ({'commence_time': 'bad'}, 'unknown_tipoff'),
    ({'commence_time': '2099-01-01T12:00:00'}, 'unknown_tipoff'),
    ({'commence_time': '2000-01-01T12:00:00Z'}, 'game_started'),
])
def test_bad_candidates_never_become_recommendations(env, changes, reason):
    db, bot = env
    refresh = MagicMock()
    sa.evaluate_and_alert(candidate(**changes), db, bot, refresh_quote=refresh)
    assert_rejected(db, bot, reason)
    refresh.assert_not_called()


@pytest.mark.parametrize('changes,reason', [
    ({'price': 1.8}, 'quote_changed'),
    ({'book': 'fanduel'}, 'quote_changed'),
    ({'checked_at': 'bad'}, 'unknown_quote_time'),
    ({'checked_at': '2000-01-01T12:00:00Z'}, 'stale_quote'),
    ({'updated_at': '2000-01-01T12:00:00Z'}, 'stale_quote'),
    ({'price': float('nan')}, 'invalid_quote'),
])
def test_bad_quotes_never_become_recommendations(env, changes, reason):
    db, bot = env
    c = candidate()
    sa.evaluate_and_alert(c, db, bot, refresh_quote=lambda _: quote(c, **changes))
    assert_rejected(db, bot, reason)


def test_no_refresher_fails_closed(env):
    db, bot = env
    sa.evaluate_and_alert(candidate(), db, bot)
    assert_rejected(db, bot, 'quote_unavailable')


def test_tipoff_is_checked_again_after_refresh(env):
    db, bot = env
    c = candidate()
    def refresh(_):
        c['commence_time'] = '2000-01-01T12:00:00Z'
        return quote(c)
    sa.evaluate_and_alert(c, db, bot, refresh_quote=refresh)
    assert_rejected(db, bot, 'game_started')


@pytest.mark.parametrize('kwargs', [{'daily_risk_cap': 0}, {'game_risk_cap': 0}])
def test_zero_risk_allowance_rejects(env, kwargs):
    db, bot = env
    c = candidate()
    sa.evaluate_and_alert(c, db, bot, refresh_quote=lambda _: quote(c), **kwargs)
    assert_rejected(db, bot, 'risk_limit')


def test_zero_stake_rejects(env, monkeypatch):
    db, bot = env
    monkeypatch.setattr(sa, 'FLAT_STAKE_PCT', 0)
    sa.evaluate_and_alert(candidate(), db, bot)
    assert_rejected(db, bot, 'nonpositive_stake')


def test_valid_candidate_uses_computed_ev_and_persists_once(env):
    db, bot = env
    c = candidate(ev=-1)  # stale field cannot suppress or justify a trade
    refresh = MagicMock(side_effect=lambda _: quote(c))
    sa.evaluate_and_alert(c, db, bot, refresh_quote=refresh)
    sa.evaluate_and_alert(c, db, bot, refresh_quote=refresh)
    with db.get_conn() as conn:
        row = conn.execute('SELECT stake, model_prob, odds FROM alerts_sent').fetchone()
        assert row['stake'] == 10
        assert conn.execute('SELECT COUNT(*) FROM alerts_sent').fetchone()[0] == 1
    assert c['ev'] == pytest.approx(0.65 * 1.9 - 1)
    assert refresh.call_count == 1
    bot.broadcast.assert_called_once()
    assert bot.broadcast.call_args.kwargs['expires_at'] > datetime.now(timezone.utc).timestamp()
    with db.get_conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM pending_alerts').fetchone()[0] == 0
    sa.record_execution.assert_called_once()


def test_fresh_quote_lookup_checks_exact_book_side_and_line():
    c = candidate()
    bridge = MagicMock()
    key = ('Test Player', 'player_rebounds', 5.5, 'UNDER', 'draftkings')
    bridge.get_props_for_game.return_value = {'quotes': {key: {'price': 1.9, 'book': 'draftkings'}}}
    refresh = sp._make_quote_refresher('g1', 123, bridge, None)
    assert quote_rejection(c, refresh(c)) is None
    assert refresh(candidate(line=6.5)) is None
    assert refresh(candidate(side='OVER')) is None
    bridge.get_props_for_game.assert_called_once()


def test_odds_api_recheck_bypasses_long_cache():
    c = candidate()
    odds = MagicMock()
    odds.get_event_odds.return_value = {'bookmakers': [{
        'title': 'draftkings', 'markets': [{'key': c['market'], 'outcomes': [
            {'description': c['player_id'], 'point': 5.5, 'name': 'Under', 'price': 1.9}]}]}]}
    refresh = sp._make_quote_refresher('g1', None, None, odds)
    assert quote_rejection(c, refresh(c)) is None
    assert odds.get_event_odds.call_args.kwargs['use_cache'] is False


def test_actionable_filter_drops_inflated_fair_edge(monkeypatch):
    monkeypatch.setattr(sp, 'OVERS_ENABLED', False)
    assert sp._actionable_edges([candidate(model_prob=0.53, odds=1.8, implied_prob=0.45)]) == []


def test_future_and_boundary_timing():
    now = datetime.now(timezone.utc)
    assert timing_rejection(candidate(commence_time=now.isoformat()), now) == 'game_started'
    assert timing_rejection(candidate(commence_time=(now + timedelta(days=1)).isoformat()), now) == 'outside_alert_window'


def test_database_refuses_expired_recommendation(env):
    db, _ = env
    result = db.insert_alert_with_limits(
        'Test Player', 'player_rebounds', 5.5, 'UNDER', 0.1, 'draftkings', 1.9, 10,
        '2026-11-15', 'g1', 'HOME', 2, 1000, 100,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    assert result is None
    with db.get_conn() as conn:
        assert conn.execute('SELECT COUNT(*) FROM alerts_sent').fetchone()[0] == 0
