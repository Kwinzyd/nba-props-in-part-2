"""Closing-line capture: sharp book first, else the alerted book's own close."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

import src.pipelines.update_clv as uc


def _tracker(track_id, player, market, line, book, odds=1.80):
    return {'track_id': track_id, 'player': player, 'market': market, 'side': 'Under',
            'alert_odds': odds, 'event_id': 'ev1', 'line': line, 'book': book,
            'game_date': '2026-11-04'}


def _book(key, title, market, player, point, price):
    return {'key': key, 'title': title, 'markets': [{'key': market, 'outcomes': [
        {'name': 'Under', 'description': player, 'point': point, 'price': price}]}]}


class _FakeDb:
    def __init__(self, rows):
        self.rows, self.updates = rows, {}

    @contextmanager
    def get_conn(self):
        rows = self.rows

        class _C:
            def execute(self, *a):
                class _R:
                    def fetchall(self_inner):
                        return rows
                return _R()
        yield _C()

    def update_clv_closing_line(self, track_id, closing, imp_c, imp_a, close_book=None):
        self.updates[track_id] = (closing, close_book)


@pytest.fixture
def run(monkeypatch):
    def _run(rows, bookmakers):
        db = _FakeDb(rows)
        tip = datetime.now(timezone.utc) + timedelta(minutes=5)

        class _Odds:
            def get_events(self):
                return [{'id': 'ev1', 'commence_time': tip.isoformat()}]

            def get_event_odds(self, **kw):
                return {'bookmakers': bookmakers}

        monkeypatch.setattr(uc, 'get_db_client', lambda: db)
        monkeypatch.setattr(uc, 'OddsApiClient', _Odds)
        monkeypatch.setattr(uc, '_fetched_events', set())
        uc.update_clv_lines()
        return db.updates
    return _run


def test_sharp_price_preferred_over_alerted_book(run):
    updates = run([_tracker(1, 'Ajay Mitchell', 'player_threes', 1.5, 'DraftKings')], [
        _book('draftkings', 'DraftKings', 'player_threes', 'Ajay Mitchell', 1.5, 1.70),
        _book('pinnacle', 'Pinnacle', 'player_threes', 'Ajay Mitchell', 1.5, 1.61),
    ])
    assert updates == {1: (1.61, 'pinnacle')}


def test_falls_back_to_alerted_book_when_no_sharp_market(run):
    # Pinnacle posts no steals; Caesars is stored by title, keyed williamhill_us.
    updates = run([_tracker(2, 'Darius Garland', 'player_steals', 1.5, 'Caesars')], [
        _book('williamhill_us', 'Caesars', 'player_steals', 'Darius Garland', 1.5, 1.40),
        _book('fanduel', 'FanDuel', 'player_steals', 'Darius Garland', 1.5, 1.50),
    ])
    assert updates == {2: (1.40, 'caesars')}


def test_no_price_at_alerted_line_records_nothing(run):
    updates = run([_tracker(3, 'Ty Jerome', 'player_rebounds', 3.5, 'betmgm')], [
        _book('betmgm', 'BetMGM', 'player_rebounds', 'Ty Jerome', 4.5, 1.90),
        _book('pinnacle', 'Pinnacle', 'player_rebounds', 'Ty Jerome', 4.5, 1.85),
    ])
    assert updates == {}
