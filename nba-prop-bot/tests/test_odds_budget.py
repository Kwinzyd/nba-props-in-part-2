"""Odds API daily credit budget, per-event cache and cost ledger."""

import json

import pytest

import src.clients.odds_api as oa


class _Resp:
    def __init__(self, cost):
        self.headers = {'x-requests-last': str(cost), 'x-requests-remaining': '400', 'x-requests-used': '100'}

    def json(self):
        return {'bookmakers': [{'key': 'pinnacle'}]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(oa, '_SPEND_PATH', str(tmp_path / 'odds_spend.json'))
    monkeypatch.setattr(oa, 'ODDS_DAILY_CREDIT_BUDGET', 20)
    monkeypatch.setattr(oa.OddsApiClient, '_odds_cache', {})
    c = oa.OddsApiClient(api_keys=['testkey123456'])
    calls = []

    def fake_execute(url, base_params, timeout=15):
        calls.append(base_params['markets'])
        return _Resp(cost=len(base_params['markets'].split(',')) - 1)  # one market "not offered"

    monkeypatch.setattr(c, '_execute_with_failover', fake_execute)
    c.calls = calls
    return c


MARKETS = ['player_points', 'player_rebounds', 'player_assists', 'player_threes']


def test_records_actual_cost_by_caller(client):
    client.get_event_odds('evt1', MARKETS, caller='scan')
    spend = oa.credits_spent_today()
    assert spend['spent'] == 3            # x-requests-last, not markets requested
    assert spend['by_caller'] == {'scan': 3}


def test_cache_hit_costs_nothing(client):
    client.get_event_odds('evt1', MARKETS, caller='scan')
    client.get_event_odds('evt1', list(reversed(MARKETS)), caller='clv')
    assert len(client.calls) == 1
    assert oa.credits_spent_today()['spent'] == 3


def test_use_cache_false_forces_fresh_fetch(client):
    client.get_event_odds('evt1', MARKETS, caller='scan')
    client.get_event_odds('evt1', MARKETS, caller='clv', use_cache=False)
    assert len(client.calls) == 2
    assert oa.credits_spent_today()['by_caller'] == {'scan': 3, 'clv': 3}


def test_budget_refuses_before_requesting(client):
    oa._record_spend(18, 'scan')           # 18 + 4 requested > 20
    with pytest.raises(oa.OddsApiBudgetError):
        client.get_event_odds('evt2', MARKETS, caller='scan')
    assert client.calls == []
    assert not client.out_of_credits      # keys aren't marked exhausted


def test_budget_error_is_quota_error_for_fallbacks():
    assert issubclass(oa.OddsApiBudgetError, oa.OddsApiQuotaError)


def test_ledger_resets_on_new_day(client):
    with open(oa._SPEND_PATH, 'w', encoding='utf-8') as fh:
        json.dump({'date': '2000-01-01', 'spent': 999, 'by_caller': {'scan': 999}}, fh)
    assert oa.credits_spent_today()['spent'] == 0
    client.get_event_odds('evt3', MARKETS, caller='scan')
    assert oa.credits_spent_today()['spent'] == 3


def test_scan_requests_only_prop_markets_by_default():
    from src.config import PROP_MARKETS, ODDS_FETCH_ALTS, GAME_MARKETS_ENABLED
    import src.pipelines.scan_props as sp
    if not ODDS_FETCH_ALTS and not GAME_MARKETS_ENABLED:
        assert sp._SHARP_FETCH_MARKETS == list(PROP_MARKETS)


def test_clv_has_its_own_allowance(client, monkeypatch):
    monkeypatch.setattr(oa, 'ODDS_CLV_CREDIT_BUDGET', 4)
    oa._record_spend(20, 'scan')           # scan budget fully spent
    client.get_event_odds('evt4', MARKETS[:2], caller='clv', use_cache=False)
    assert oa.credits_spent_today()['by_caller']['clv'] == 1
    with pytest.raises(oa.OddsApiBudgetError):
        client.get_event_odds('evt5', MARKETS, caller='clv')   # 1 + 4 requested > 4
    with pytest.raises(oa.OddsApiBudgetError):
        client.get_event_odds('evt6', MARKETS[:1], caller='scan')


def test_clv_spend_does_not_count_against_scan_budget(client, monkeypatch):
    monkeypatch.setattr(oa, 'ODDS_CLV_CREDIT_BUDGET', 50)
    oa._record_spend(15, 'clv')
    client.get_event_odds('evt7', MARKETS, caller='scan')   # 0 scan spent + 4 <= 20
    assert oa.credits_spent_today()['by_caller']['scan'] == 3
