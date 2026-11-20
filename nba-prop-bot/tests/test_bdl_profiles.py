"""BDL season profiles are skipped (no API calls) unless BDL_PROFILES_ENABLED."""

import src.clients.bdl_bridge as bb


class _NoCallsBDL:
    def __getattr__(self, name):
        raise AssertionError(f"unexpected BDL call: {name}")


def test_profiles_disabled_returns_defaults_without_calls(monkeypatch):
    monkeypatch.setattr(bb, 'BDL_PROFILES_ENABLED', False)
    bridge = bb.BDLBridge(_NoCallsBDL())
    bridge.prefetch_player_profiles([1, 2, 3], season=2026)
    prof = bridge.get_player_season_profile(1, season=2026)
    assert prof['real_usage_pct'] == 0.0 and prof['pnr_bh_freq'] == 0.0
