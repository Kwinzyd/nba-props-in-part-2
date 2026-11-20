"""Shadow overs: paper picks selected on the raw model prob, stored apart, graded."""

import sqlite3
from contextlib import contextmanager

import pytest

import src.pipelines.shadow_overs as so
from src.pipelines.settle_results import _grade_prop


class _Db:
    """Minimal sqlite-backed client with the shadow_picks table from schema.sql."""

    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        with open('src/data/schema.sql', encoding='utf-8') as fh:
            ddl = fh.read()
        start = ddl.index('CREATE TABLE IF NOT EXISTS shadow_picks')
        self.conn.execute(ddl[start:ddl.index(';', start)])

    @contextmanager
    def get_conn(self):
        yield self.conn
        self.conn.commit()


def _cand(side='OVER', raw=0.62, cal=0.53, odds=1.90, hours=2.0, player='Jalen Green'):
    return {'side': side, 'raw_model_prob': raw, 'model_prob': cal, 'odds': odds,
            'implied_prob': 1 / odds, 'hours_to_tipoff': hours, 'market': 'player_points',
            'line': 20.5, 'player_id': player, 'book': 'draftkings',
            'game_date': '2026-11-08', 'event_id': 'ev1', 'projected_minutes': 32,
            'mean': 0, 'calibrated': True}


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(so, 'OVERS_ENABLED', False)
    monkeypatch.setattr(so, 'SHADOW_OVERS_ENABLED', True)
    return _Db()


def test_selects_overs_on_raw_prob_only(db):
    cands = [_cand(), _cand(side='UNDER', player='U'), _cand(raw=0.53, player='Weak'),
             _cand(hours=6.0, player='Too early')]
    assert so.record_shadow_overs(cands, db) == 1
    row = db.conn.execute('SELECT * FROM shadow_picks').fetchone()
    assert row['player_name'] == 'Jalen Green' and row['side'] == 'OVER'
    assert row['raw_model_prob'] == pytest.approx(0.62) and row['model_prob'] == pytest.approx(0.53)
    assert cands[0]['model_prob'] == 0.53          # live candidate untouched


def test_first_sighting_kept(db):
    so.record_shadow_overs([_cand(odds=1.90)], db)
    assert so.record_shadow_overs([_cand(odds=2.10)], db) == 0
    assert db.conn.execute('SELECT odds FROM shadow_picks').fetchone()['odds'] == pytest.approx(1.90)


def test_off_when_overs_live(db, monkeypatch):
    monkeypatch.setattr(so, 'OVERS_ENABLED', True)
    assert so.record_shadow_overs([_cand()], db) == 0


def test_settle_and_report(db):
    so.record_shadow_overs([_cand(), _cand(player='Ja Morant')], db)
    stats = {'Jalen Green': {'pts': 25}, 'Ja Morant': None}
    assert so.unsettled_shadow_dates(db, '2026-11-01', '2026-11-08') == ['2026-11-08']
    assert so.settle_shadow_picks(db, '2026-11-08', stats.get, _grade_prop) == 1   # Morant DNP
    report = so.shadow_report(db)
    assert report['ALL']['n'] == 1 and report['ALL']['wins'] == 1
