"""Sharp picks: Sharp Alerts stored on first sighting, graded, reported; DNPs void."""

import sqlite3
from contextlib import contextmanager

import pytest

import src.pipelines.shadow_overs as so
from src.pipelines.settle_results import _grade_prop
from src.pipelines.sharp_picks import record_sharp_pick
from src.pipelines.scan_props import _rec_bookmakers


class _Db:
    """Minimal sqlite-backed client with the sharp_picks table from schema.sql."""

    def __init__(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        with open('src/data/schema.sql', encoding='utf-8') as fh:
            ddl = fh.read()
        start = ddl.index('CREATE TABLE IF NOT EXISTS sharp_picks')
        self.conn.execute(ddl[start:ddl.index(';', start)])

    @contextmanager
    def get_conn(self):
        yield self.conn
        self.conn.commit()


def _pick(db, player='Jalen Green', side='Under', odds=2.05, fair=0.53):
    return record_sharp_pick(
        db, game_date='2026-11-08', event_id='ev1', player_name=player,
        market='player_points', line=20.5, side=side, book='FanDuel', odds=odds,
        fair_prob=fair, consensus_label='Pinnacle', hours_to_tipoff=2.0)


def test_first_sighting_kept():
    db = _Db()
    assert _pick(db, odds=2.05) is True
    assert _pick(db, odds=2.20) is False           # later scan, same prop
    row = db.conn.execute('SELECT * FROM sharp_picks').fetchone()
    assert row['side'] == 'UNDER' and row['odds'] == pytest.approx(2.05)
    assert row['edge'] == pytest.approx(0.53 - 1 / 2.05)


def test_settle_and_report():
    db = _Db()
    _pick(db)
    _pick(db, player='Ja Morant')
    _pick(db, player='Benched Guy')
    stats = {'Jalen Green': {'pts': 18, 'min': '33'},
             'Benched Guy': {'pts': 0, 'min': '00'},   # DNP → void, not an under win
             'Ja Morant': None}                         # game not final
    assert so.unsettled_shadow_dates(db, '2026-11-01', '2026-11-08', table='sharp_picks') == ['2026-11-08']
    assert so.settle_shadow_picks(db, '2026-11-08', stats.get, _grade_prop, table='sharp_picks') == 2
    benched = db.conn.execute("SELECT won, push FROM sharp_picks WHERE player_name='Benched Guy'").fetchone()
    assert benched['push'] and not benched['won']
    report = so.shadow_report(db, table='sharp_picks')
    assert report['ALL']['n'] == 1 and report['ALL']['wins'] == 1


def test_unknown_table_rejected():
    with pytest.raises(ValueError):
        so.shadow_report(_Db(), table='alerts_sent')


@pytest.mark.parametrize('minutes,won', [('00', None), ('0:00', None), ('', None), (None, None),
                                         ('12:30', 1), ('25', 1)])
def test_dnp_is_void(minutes, won):
    assert _grade_prop('player_threes', 1.5, 'UNDER', {'fg3m': 0, 'min': minutes})[0] == won


def test_rows_without_minutes_still_grade():
    assert _grade_prop('player_points', 20.5, 'OVER', {'pts': 25}) == (1, 25.0)


def test_rec_bookmakers_only_dk_fd():
    books = [{'key': 'pinnacle', 'title': 'Pinnacle'}, {'key': 'draftkings', 'title': 'DraftKings'},
             {'key': 'fanduel', 'title': 'FanDuel'}, {'key': 'betrivers', 'title': 'BetRivers'}]
    assert [b['key'] for b in _rec_bookmakers(books)] == ['draftkings', 'fanduel']
