"""Watchdog stall clocks restart from a wake-up, so PC sleep isn't a 'hang'."""

import src.pipelines.run_scheduler as rs


def test_stall_measured_from_wake(monkeypatch):
    monkeypatch.setattr(rs, '_last_tick', 1_000.0)
    monkeypatch.setattr(rs, '_scan_in_progress', True)
    monkeypatch.setattr(rs, '_scan_started_at', 900.0)
    # 10 h asleep, checked 60 s after waking: nothing looks stalled.
    now = 1_000.0 + 10 * 3600
    stale, scan_age = rs._stall_ages(now, resumed_at=now - 60)
    assert stale == 60 and scan_age == 60
    assert stale < rs._WATCHDOG_EXIT_SEC and scan_age < rs._SCAN_STALL_SEC


def test_real_hang_still_detected(monkeypatch):
    monkeypatch.setattr(rs, '_last_tick', 1_000.0)
    monkeypatch.setattr(rs, '_scan_in_progress', True)
    monkeypatch.setattr(rs, '_scan_started_at', 1_000.0)
    now = 1_000.0 + 2 * 3600          # awake the whole time
    stale, scan_age = rs._stall_ages(now, resumed_at=0.0)
    assert stale > rs._WATCHDOG_EXIT_SEC and scan_age > rs._SCAN_STALL_SEC


def test_no_scan_running(monkeypatch):
    monkeypatch.setattr(rs, '_last_tick', 1_000.0)
    monkeypatch.setattr(rs, '_scan_in_progress', False)
    assert rs._stall_ages(1_030.0, 0.0) == (30.0, 0.0)
