"""Tests for calibration_model.py"""

from src.models.calibration_model import calibrate_prob, calibrate_edge_candidate


def test_extremes_preserved():
    """Well-calibrated extremes should pass through roughly unchanged."""
    # >65% bucket: 81.7% predicted, 81.8% actual
    assert abs(calibrate_prob(0.82) - 0.82) < 0.02, \
        "High-confidence bucket should be near-unchanged"

    # <45% bucket: 41.8% predicted, 40.0% actual
    low = calibrate_prob(0.42)
    assert abs(low - 0.40) < 0.03, \
        f"Low bucket should calibrate near 40%, got {low:.1%}"


def test_overconfident_middle_compressed():
    """The 55-65% band should be pulled down significantly."""
    # 58% model → should calibrate to ~48% (actual was 46.2%)
    cal_58 = calibrate_prob(0.58)
    assert cal_58 < 0.52, \
        f"58% model prob should calibrate below 52%, got {cal_58:.1%}"

    # 63% model → should calibrate to ~50% (actual was 50.0%)
    cal_63 = calibrate_prob(0.63)
    assert cal_63 < 0.55, \
        f"63% model prob should calibrate below 55%, got {cal_63:.1%}"


def test_calibration_monotonic_at_extremes():
    """Higher raw prob should generally give higher calibrated prob outside the problem zone."""
    # Below 50% and above 70% should be monotonic
    assert calibrate_prob(0.30) < calibrate_prob(0.40)
    assert calibrate_prob(0.75) < calibrate_prob(0.85)
    assert calibrate_prob(0.85) < calibrate_prob(0.95)


def test_calibrate_edge_candidate_preserves_raw():
    """calibrate_edge_candidate should store the raw value and overwrite model_prob."""
    c = {'model_prob': 0.60, 'implied_prob': 0.50}
    calibrate_edge_candidate(c)

    assert c['raw_model_prob'] == 0.60, "Raw prob should be preserved"
    assert c['model_prob'] != 0.60, "model_prob should be overwritten with calibrated value"
    assert c['calibrated'] is True


def test_boundary_values():
    """Edge cases: 0, 1, and near-boundary values."""
    assert calibrate_prob(0.0) >= 0.0
    assert calibrate_prob(1.0) <= 1.0
    assert 0.0 < calibrate_prob(0.01) < 1.0
    assert 0.0 < calibrate_prob(0.99) <= 1.0


# ── Market-anchored calibration ──────────────────────────────────────────────

import src.models.calibration_model as _cm

_PARAMS = {'k': 0.0, 'b_under': 0.45}


def _with_anchor(monkeypatch, params=_PARAMS):
    monkeypatch.setattr(_cm, '_get_anchor', lambda: params)


def test_anchored_over_with_k0_returns_market_price():
    assert abs(_cm.anchored_prob(0.80, 0.55, False, _PARAMS) - 0.55) < 1e-9


def test_anchored_under_bias_raises_prob():
    p = _cm.anchored_prob(0.80, 0.55, True, _PARAMS)
    assert 0.64 < p < 0.68, p


def test_anchored_k1_returns_raw():
    assert abs(_cm.anchored_prob(0.70, 0.50, False, {'k': 1.0, 'b_under': 0.0}) - 0.70) < 1e-9


def test_candidate_uses_book_odds_and_side(monkeypatch):
    _with_anchor(monkeypatch)
    c = calibrate_edge_candidate({'model_prob': 0.80, 'odds': 1.80, 'side': 'UNDER'})
    assert c['raw_model_prob'] == 0.80
    assert abs(c['model_prob'] - _cm.anchored_prob(0.80, 1 / 1.80, True, _PARAMS)) < 1e-9


def test_candidate_below_min_raw_edge_gets_no_edge(monkeypatch):
    _with_anchor(monkeypatch)
    implied = 1 / 1.80
    c = calibrate_edge_candidate({'model_prob': implied + 0.01, 'odds': 1.80, 'side': 'UNDER'})
    assert abs(c['model_prob'] - implied) < 1e-9


def test_no_odds_falls_back_to_legacy(monkeypatch):
    _with_anchor(monkeypatch)
    assert calibrate_prob(0.6) == _cm.calibrate_prob(0.6, implied=None)
