"""keep_system_awake asks Windows for ES_CONTINUOUS | ES_SYSTEM_REQUIRED only."""

import sys
import types

import src.utils.keep_awake as ka


def test_requests_system_required(monkeypatch):
    calls = []
    kernel32 = types.SimpleNamespace(SetThreadExecutionState=lambda f: calls.append(f) or 0x80000000)
    import ctypes
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=kernel32), raising=False)
    assert ka.keep_system_awake() is True
    assert calls == [0x80000001]           # no ES_DISPLAY_REQUIRED: screen may still sleep


def test_rejected_call_reports_false(monkeypatch):
    import ctypes
    kernel32 = types.SimpleNamespace(SetThreadExecutionState=lambda f: 0)
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(ctypes, 'windll', types.SimpleNamespace(kernel32=kernel32), raising=False)
    assert ka.keep_system_awake() is False


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'linux')
    assert ka.keep_system_awake() is False
