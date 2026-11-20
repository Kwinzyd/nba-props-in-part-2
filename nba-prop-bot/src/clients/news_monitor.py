"""
Real-time breaking injury news monitor via RSS feeds.

Polls Rotowire Basketball and ESPN NBA RSS every NEWS_POLL_INTERVAL seconds.
On first poll, silently marks all existing items as seen (no false alerts on restart).
On subsequent polls, only NEW items matching injury keywords are returned.
"""
import re
import feedparser
from datetime import datetime
from typing import List, Dict, Set
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

FEED_URLS = [
    "https://www.rotowire.com/basketball/rss/news.xml",
    "https://www.espn.com/espn/rss/nba/news",
]

INJURY_KEYWORDS = {
    'out', 'injured', 'injury', 'doubtful', 'questionable',
    'ankle', 'knee', 'hip', 'shoulder', 'concussion', 'illness',
    'ruled out', 'day-to-day', 'dtd', 'inactive', 'scratched',
    "won't play", 'will not play', 'listed out', 'expected to miss',
    'placed on', 'left the game',
}


def mentions(text_lower: str, phrase: str) -> bool:
    """Whole-word match, so 'out' doesn't fire on 'checking out' or 'about'."""
    return phrase in text_lower and re.search(
        r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])', text_lower) is not None


def find_players(text_lower: str, names) -> List[str]:
    return [n for n in names if mentions(text_lower, n)]


class BreakingNewsMonitor:
    def __init__(self):
        self._seen_ids: Set[str] = set()
        self._initialized: bool = False   # first poll marks items seen, no alerts
        self._player_names: Set[str] = self._load_player_names()

    @staticmethod
    def _load_player_names() -> Set[str]:
        """Load full NBA player name set for O(1) mention detection."""
        try:
            from nba_api.stats.static import players
            from src.clients.nba_stats import get_current_roster
            names = {p['full_name'].lower() for p in players.get_players()}
            # Rookies / new signings aren't in the bundled static list
            names |= {n.lower() for n in get_current_roster()['id_to_name'].values()}
            return names
        except Exception:
            return set()

    @staticmethod
    def _is_injury_related(title: str, summary: str) -> bool:
        text = (title + ' ' + summary).lower()
        return any(mentions(text, kw) for kw in INJURY_KEYWORDS)

    def _extract_player_names(self, title: str, summary: str) -> List[str]:
        return find_players((title + ' ' + summary).lower(), self._player_names)

    def get_breaking_injuries(self) -> List[Dict]:
        """
        Poll all feeds. Returns NEW injury items as list of dicts:
            {player_name, all_players, title, summary, source, published_at}
        Returns [] on first poll (initialisation pass).
        """
        new_items = []
        for url in FEED_URLS:
            try:
                feed = feedparser.parse(url)
                source = feed.feed.get('title', url)
                for entry in feed.entries:
                    entry_id = entry.get('id') or entry.get('link', '')
                    if entry_id in self._seen_ids:
                        continue
                    self._seen_ids.add(entry_id)
                    if not self._initialized:
                        continue     # first pass: mark as seen, don't alert
                    title   = entry.get('title',   '')
                    summary = entry.get('summary', '')
                    if not self._is_injury_related(title, summary):
                        continue
                    players_found = self._extract_player_names(title, summary)
                    if not players_found:
                        continue     # injury words but no NBA player named
                    new_items.append({
                        'player_name':  players_found[0].title() if players_found else '',
                        'all_players':  [p.title() for p in players_found],
                        'title':        title,
                        'summary':      summary[:300],
                        'source':       source,
                        'published_at': entry.get('published', datetime.now().strftime('%H:%M ET')),
                    })
            except Exception as e:
                logger.warning(f"News feed error ({url}): {e}")

        self._initialized = True
        return new_items
