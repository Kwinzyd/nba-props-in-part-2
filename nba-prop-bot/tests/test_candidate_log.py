"""Candidate log (every scanned prop, refreshed until settled), DK/FD-only best price, sharp line storage."""

import sqlite3
from contextlib import contextmanager

import pytest

import src.pipelines.candidate_log as cl
import src.pipelines.scan_props as sp
import src.pipelines.shadow_overs as so
from src.pipelines.settle_results import _grade_prop


class _Db:
    """Minimal sqlite-backed client with the scan_candidates table from schema.sql."""

    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        with open('src/data/schema.sql', encoding='utf-8') as fh:
            ddl = fh.read()
        start = ddl.index('CREATE TABLE IF NOT EXISTS scan_candidates')
        self.conn.execute(ddl[start:ddl.index(';', start)])
        self.line_history = []

    @contextmanager
    def get_conn(self):
        yield self.conn
        self.conn.commit()

    def insert_line_history_batch(self, records):
        self.line_history.extend(records)


def _cand(player='Jalen Green', side='UNDER', odds=1.91, raw=0.62, cal=0.55, implied=0.50, **kw):
    c = {'player_id': player, 'market': 'player_points_alternate', 'line': 20.5, 'side': side,
         'odds': odds, 'book': 'DraftKings', 'implied_prob': implied, 'raw_model_prob': raw,
         'model_prob': cal, 'mean': 19.0, 'projected_minutes': 32.0, 'game_date': '2026-11-08',
         'event_id': 'ev1', 'hours_to_tipoff': 2.0, 'consensus_prob': None}
    c.update(kw)
    return c


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(cl, 'CANDIDATE_LOG_ENABLED', True)
    return _Db()


def test_records_base_market_and_refreshes_until_settled(db):
    assert cl.record_candidates([_cand(), _cand(side='OVER', odds=1.91, implied=0.50)], db) == 2
    cl.record_candidates([_cand(odds=1.80, consensus_prob=0.45)], db)       # later scan
    row = db.conn.execute("SELECT * FROM scan_candidates WHERE side='UNDER'").fetchone()
    assert row['market'] == 'player_points' and row['odds'] == pytest.approx(1.80)
    assert row['sharp_over_prob'] == pytest.approx(0.45)
    cl.record_candidates([_cand(odds=2.00)], db)                            # sharp prob kept
    assert db.conn.execute("SELECT sharp_over_prob FROM scan_candidates WHERE side='UNDER'").fetchone()[0] \
        == pytest.approx(0.45)
    assert db.conn.execute('SELECT COUNT(*) FROM scan_candidates').fetchone()[0] == 2

    so.settle_shadow_picks(db, '2026-11-08', {'Jalen Green': {'pts': 18, 'min': '30'}}.get,
                           _grade_prop, table='scan_candidates')
    cl.record_candidates([_cand(odds=1.50)], db)                            # after settlement
    row = db.conn.execute("SELECT odds, won FROM scan_candidates WHERE side='UNDER'").fetchone()
    assert row['odds'] == pytest.approx(2.00) and row['won']


def test_skips_rows_without_price_or_player(db):
    assert cl.record_candidates([_cand(odds=0), _cand(player=None)], db) == 0


def test_off_switch(db, monkeypatch):
    monkeypatch.setattr(cl, 'CANDIDATE_LOG_ENABLED', False)
    assert cl.record_candidates([_cand()], db) == 0


def test_model_eval_scores_model_against_market(db):
    # Model says 0.62 on every under; under wins 1 of 2 → market (0.50) scores better.
    cl.record_candidates([_cand(), _cand(player='Ja Morant')], db)
    so.settle_shadow_picks(db, '2026-11-08', {'Jalen Green': {'pts': 18, 'min': '30'},
                                              'Ja Morant': {'pts': 25, 'min': '30'}}.get,
                           _grade_prop, table='scan_candidates')
    r = cl.model_eval_report(db)['ALL']
    assert r['n'] == 2 and r['brier_market'] < r['brier_model']
    assert r['picks'] == 2 and r['pick_roi'] == pytest.approx((0.91 - 1) / 2)


def test_best_rec_odds_ignores_other_books():
    recs = [('A', 'player_points', 'betmgm', 20.5, 'UNDER', 2.10, 0.48),
            ('A', 'player_points', 'draftkings', 20.5, 'UNDER', 1.87, 0.53),
            ('A', 'player_points', 'fanduel', 20.5, 'UNDER', 1.91, 0.52)]
    assert sp._best_rec_odds(recs) == {('A', 'player_points', 20.5, 'UNDER'): {'price': 1.91, 'book': 'fanduel'}}


def test_sharp_lines_stored_once_per_price_change():
    def book(key, price):
        return {'key': key, 'title': key.title(), 'markets': [{'key': 'player_points', 'outcomes': [
            {'name': 'Under', 'description': 'A', 'point': 20.5, 'price': price},
            {'name': 'Over', 'description': 'A', 'point': 20.5, 'price': 1.95}]}]}
    db = _Db()
    sp._SHARP_LH_SEEN.clear()
    sp._store_sharp_lines(db, 'ev1', [book('pinnacle', 1.90), book('draftkings', 1.87)])
    assert db.line_history == [('A', 'player_points', 'pinnacle', 20.5, 'UNDER', 1.90, 1 / 1.90),
                               ('A', 'player_points', 'pinnacle', 20.5, 'OVER', 1.95, 1 / 1.95)]
    sp._store_sharp_lines(db, 'ev1', [book('pinnacle', 1.90)])     # cached response: skipped
    assert len(db.line_history) == 2
    sp._store_sharp_lines(db, 'ev1', [book('pinnacle', 1.85)])     # moved: stored
    assert len(db.line_history) == 4
