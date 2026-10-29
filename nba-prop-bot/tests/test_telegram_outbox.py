"""Failed Telegram sends are queued in the outbox and redelivered later."""

import time

import pytest
import requests

import src.clients.telegram_bot as tb


class _Resp:
    def __init__(self, status):
        self.status_code = status
        self.ok = status == 200
        self.text = '{"ok": false}'


@pytest.fixture
def bot(tmp_path, monkeypatch):
    monkeypatch.setattr(tb, '_OUTBOX_PATH', str(tmp_path / 'outbox.json'))
    monkeypatch.setattr(tb, '_last_failure', 0.0)
    b = tb.TelegramBotClient(token='123:abc', chat_id='42')
    b.sent = []
    b.mode = 'timeout'

    def fake_post(url, json=None, timeout=None):
        if b.mode == 'timeout':
            raise requests.exceptions.ReadTimeout()
        if b.mode == 'bad_request':
            return _Resp(400)
        b.sent.append(json['text'])
        return _Resp(200)

    monkeypatch.setattr(tb.requests, 'post', fake_post)
    return b


def test_timeout_queues_message(bot):
    assert bot.send_instant('<b>play 1</b>') is False
    assert tb.outbox_size() == 1


def test_cooldown_skips_network_and_queues(bot):
    bot.send_instant('play 1')
    bot.mode = 'ok'                       # would succeed, but cooldown is active
    assert bot.send_instant('play 2') is False
    assert bot.sent == [] and tb.outbox_size() == 2


def test_flush_delivers_in_order_marked_delayed(bot):
    bot.send_instant('play 1')
    bot.send_instant('play 2')
    assert bot.flush_outbox() == 0        # still down: nothing delivered, nothing lost
    assert tb.outbox_size() == 2
    bot.mode = 'ok'
    assert bot.flush_outbox() == 2
    assert tb.outbox_size() == 0
    assert [s.split('\n', 1)[1] for s in bot.sent] == ['play 1', 'play 2']
    assert all('Delayed' in s.split('\n', 1)[0] for s in bot.sent)
    assert bot.send_instant('play 3') is True   # cooldown cleared after recovery


def test_rejected_message_is_not_queued(bot):
    bot.mode = 'bad_request'
    assert bot.send_instant('<b>broken html') is False
    assert tb.outbox_size() == 0


def test_stale_messages_dropped(bot):
    bot.send_instant('old play')
    items = tb._load_outbox()
    items[0]['queued_at'] = time.time() - tb._OUTBOX_MAX_AGE_SEC - 1
    tb._save_outbox(items)
    bot.mode = 'ok'
    assert bot.flush_outbox() == 0
    assert tb.outbox_size() == 0 and bot.sent == []
