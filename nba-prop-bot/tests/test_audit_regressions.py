"""Regression coverage for failures observed in the November production audit."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.data.db import DatabaseClient
from src.pipelines import breaking_news, calibration, drift_monitor, send_alerts, settle_results


@pytest.fixture
def db(tmp_path):
    return DatabaseClient(str(tmp_path / 'audit.db'))


def test_news_feed_html_is_safe_for_telegram(monkeypatch):
    monitor, bot = MagicMock(), MagicMock()
    monitor.get_breaking_injuries.return_value = [{
        'title': 'Player <em>OUT</em> & questionable',
        'summary': '<p>Coach says <a href="https://example.com">rest</a> & recovery</p>',
        'source': 'RSS & news', 'published_at': 'today',
    }]
    monkeypatch.setattr('src.data.db.get_db_client', lambda: None)
    assert breaking_news.check_breaking_news(monitor, bot)
    message = bot.broadcast.call_args.args[0]
    assert '<p>' not in message and '<em>' not in message and '<a ' not in message
    assert 'rest &amp; recovery' in message and '<b>BREAKING:' in message


def test_drift_snapshot_upserts_and_excludes_today_from_baseline(db):
    drift_monitor._write_snapshot(db, 'all_time', 'all', 0.25, 100)
    drift_monitor._write_snapshot(db, 'all_time', 'all', 0.22, 120)
    with db.get_conn() as conn:
        rows = conn.execute('SELECT brier, n_samples FROM model_health').fetchall()
    assert len(rows) == 1 and rows[0]['brier'] == pytest.approx(0.22)
    assert rows[0]['n_samples'] == 120
    assert drift_monitor._historical_brier_stats(db)[2] == 0


def test_drift_uses_probability_delivered_in_alert(db):
    alert_id = db.insert_alert('Player', 'player_points', 10.5, 'UNDER',
                               0.1, 'draftkings', 2.0, raw_model_prob=0.9, model_prob=0.62)
    with db.get_conn() as conn:
        conn.execute('INSERT INTO bet_results (alert_id, actual_result, won, push) VALUES (?, 5, 1, 0)', (alert_id,))
    assert drift_monitor._fetch_settled(db) == [(0.62, 1)]


def test_calibration_counts_alerts_once_and_excludes_pushes(db, monkeypatch, capsys):
    for push in (False, True):
        alert_id = db.insert_alert('Player', 'player_points', 10.5, 'UNDER',
                                   0.1, 'draftkings', 2.0, model_prob=0.7)
        with db.get_conn() as conn:
            conn.execute('INSERT INTO bet_results (alert_id, actual_result, won, push) VALUES (?, 5, 1, ?)', (alert_id, push))
    monkeypatch.setattr(calibration, 'get_db_client', lambda: db)
    calibration.check_calibration()
    output = capsys.readouterr().out
    assert 'Total Graded Bets: 1' in output and '0.0900' in output


def test_settlement_queries_work_on_sqlite(db, monkeypatch):
    target = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    alert_id = db.insert_alert('Test Player', 'player_points', 10.5, 'UNDER',
                               0.1, 'draftkings', 2.0, game_date=target)
    bdl, bot = MagicMock(), MagicMock()
    bdl.get_games_by_date.return_value = [{'id': 123, 'status': 'Final'}]
    bdl.get_game_stats.return_value = [{
        'player': {'first_name': 'Test', 'last_name': 'Player'},
        'game': {'id': 123}, 'pts': 5, 'min': '25:00',
    }]
    monkeypatch.setattr(settle_results, 'get_db_client', lambda: db)
    monkeypatch.setattr(settle_results, 'BDLClient', lambda: bdl)
    monkeypatch.setattr(settle_results, 'TelegramBotClient', lambda: bot)
    settle_results.settle_recent()
    with db.get_conn() as conn:
        row = conn.execute('SELECT won, push FROM bet_results WHERE alert_id = ?', (alert_id,)).fetchone()
    assert row['won'] and not row['push']
    settle_results.settle_recent()
    assert bdl.get_game_stats.call_count == 1


def test_parlay_rounding_never_exceeds_cap(monkeypatch):
    monkeypatch.setattr(send_alerts, 'BANKROLL', 100.0)
    stake = send_alerts._parlay_kelly_stake([{'event_id': 'g1'}], 0.8, 2.0, max_pct=0.001)
    assert 0 < stake <= 0.1
