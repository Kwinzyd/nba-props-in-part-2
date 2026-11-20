"""Overs are dropped from actionable edges unless OVERS_ENABLED."""

import src.pipelines.scan_props as sp


def _cands():
    return [
        {'player_id': 'A', 'market': 'player_points', 'side': 'OVER', 'edge': 0.12,
         'edge_min_applied': 0.08, 'hours_to_tipoff': 1.0, 'odds': 1.9, 'model_prob': 0.65, 'line': 20.5},
        {'player_id': 'B', 'market': 'player_points', 'side': 'UNDER', 'edge': 0.12,
         'edge_min_applied': 0.08, 'hours_to_tipoff': 1.0, 'odds': 1.9, 'model_prob': 0.65, 'line': 20.5},
    ]


def test_overs_dropped_by_default(monkeypatch):
    monkeypatch.setattr(sp, 'OVERS_ENABLED', False)
    monkeypatch.setattr(sp, 'rank_edges', lambda c: c)
    assert [e['side'] for e in sp._actionable_edges(_cands())] == ['UNDER']


def test_overs_kept_when_enabled(monkeypatch):
    monkeypatch.setattr(sp, 'OVERS_ENABLED', True)
    monkeypatch.setattr(sp, 'rank_edges', lambda c: c)
    assert sorted(e['side'] for e in sp._actionable_edges(_cands())) == ['OVER', 'UNDER']
