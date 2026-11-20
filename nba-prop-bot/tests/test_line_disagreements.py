"""Sharp line-disagreement alerts: DraftKings/FanDuel only, one per player per day."""
from unittest.mock import MagicMock

import pytest

from src.data.db import DatabaseClient
from src.pipelines import scan_props


def _pinnacle(player, market, line):
    return {'title': 'Pinnacle', 'markets': [{'key': market, 'outcomes': [
        {'description': player, 'point': line, 'name': 'Over', 'price': 1.91},
        {'description': player, 'point': line, 'name': 'Under', 'price': 1.91},
    ]}]}


def _rec(player, market, book, line, side, price):
    return (player, market, book, line, side, price, 1.0 / price)


@pytest.fixture
def db(tmp_path):
    return DatabaseClient(str(tmp_path / 'test.db'))


def _run(db, records, sharp):
    bot = MagicMock()
    scan_props.detect_line_disagreements(
        records, sharp, 'Home', 'Away', '2026-11-16', 'evt1', db=db, bot=bot)
    return [c.args[0] for c in bot.broadcast.call_args_list]


def test_only_dk_fd_one_alert_per_player_best_gap_and_price(db):
    sharp = [_pinnacle('Jayson Tatum', 'player_rebounds', 6.5),
             _pinnacle('Jayson Tatum', 'player_points_rebounds_assists', 39.5)]
    records = [
        _rec('Jayson Tatum', 'player_rebounds', 'Caesars', 8.5, 'UNDER', 2.00),
        _rec('Jayson Tatum', 'player_rebounds', 'DraftKings', 7.5, 'UNDER', 1.87),
        _rec('Jayson Tatum', 'player_rebounds', 'FanDuel', 7.5, 'UNDER', 1.91),
        _rec('Jayson Tatum', 'player_points_rebounds_assists', 'FanDuel', 40.5, 'UNDER', 1.85),
    ]
    sent = _run(db, records, sharp)
    assert len(sent) == 1
    assert 'Fanduel' in sent[0] and 'UNDER 7.5' in sent[0] and '1.91' in sent[0]
    assert 'Caesars' not in sent[0]


def test_bigger_gap_beats_better_price(db):
    sharp = [_pinnacle('Donovan Mitchell', 'player_points', 25.5),
             _pinnacle('Donovan Mitchell', 'player_assists', 5.5)]
    records = [
        _rec('Donovan Mitchell', 'player_points', 'DraftKings', 27.5, 'UNDER', 1.80),
        _rec('Donovan Mitchell', 'player_assists', 'FanDuel', 6.5, 'UNDER', 2.00),
    ]
    sent = _run(db, records, sharp)
    assert len(sent) == 1 and 'UNDER 27.5' in sent[0]


def test_not_resent_on_later_scan(db):
    sharp = [_pinnacle('Paul George', 'player_threes', 3.5)]
    records = [_rec('Paul George', 'player_threes', 'DraftKings', 2.5, 'OVER', 1.80)]
    assert len(_run(db, records, sharp)) == 1
    assert _run(db, records, sharp) == []


def test_skipped_when_side_not_priced_at_soft_book(db):
    sharp = [_pinnacle('Sam Hauser', 'player_points', 7.5)]
    # Only the UNDER is listed, but the gap calls for the OVER.
    records = [_rec('Sam Hauser', 'player_points', 'DraftKings', 6.5, 'UNDER', 1.91)]
    assert _run(db, records, sharp) == []


def test_betmgm_only_disagreement_not_sent(db):
    sharp = [_pinnacle('Josh Giddey', 'player_points', 12.5)]
    records = [_rec('Josh Giddey', 'player_points', 'BetMGM', 13.5, 'UNDER', 1.91)]
    assert _run(db, records, sharp) == []
