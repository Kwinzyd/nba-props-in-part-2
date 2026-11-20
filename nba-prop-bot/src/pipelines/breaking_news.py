from html import escape

from bs4 import BeautifulSoup

from src.clients.telegram_bot import TelegramBotClient
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def check_breaking_news(monitor, bot: TelegramBotClient) -> bool:
    """
    Poll for new breaking injury news (Twitter/Nitter or RSS fallback).
    Sends a Telegram alert for each new item.
    Returns True if any new items were found (caller should trigger immediate scan).
    """
    items = monitor.get_breaking_injuries()
    if not items:
        return False

    for item in items:
        # RSS summaries contain arbitrary HTML, which Telegram cannot render.
        # Strip feed markup, then escape all external fields inside our HTML.
        fields = {
            key: escape(BeautifulSoup(str(item.get(key, '')), 'html.parser').get_text(' ', strip=True))
            for key in ('title', 'summary', 'source', 'published_at')
        }
        source_icon = "🐦" if item['source'].startswith("@") else "📰"
        msg = (
            f"⚡ <b>BREAKING: {fields['title']}</b>\n\n"
            f"{fields['summary']}\n\n"
            f"{source_icon} {fields['source']} · {fields['published_at']}\n"
            f"<i>→ Triggering immediate prop scan...</i>"
        )
        from src.data.db import get_db_client
        bot.broadcast(msg, db=get_db_client())
        logger.info(f"Breaking news alert sent: {item['title']}")

    return True
