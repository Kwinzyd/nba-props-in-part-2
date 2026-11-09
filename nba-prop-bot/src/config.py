import os
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEYS_RAW = os.getenv("ODDS_API_KEYS", "")
ODDS_API_KEYS = [k.strip() for k in ODDS_API_KEYS_RAW.split(",") if k.strip()]
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEYS and ODDS_API_KEY:
    ODDS_API_KEYS = [ODDS_API_KEY]
elif ODDS_API_KEYS and not ODDS_API_KEY:
    ODDS_API_KEY = ODDS_API_KEYS[0]

SPORTRADAR_API_KEYS_RAW = os.getenv("SPORTRADAR_API_KEYS", "")
SPORTRADAR_API_KEYS = [k.strip() for k in SPORTRADAR_API_KEYS_RAW.split(",") if k.strip()]
SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY")
if not SPORTRADAR_API_KEYS and SPORTRADAR_API_KEY:
    SPORTRADAR_API_KEYS = [SPORTRADAR_API_KEY]
elif SPORTRADAR_API_KEYS and not SPORTRADAR_API_KEY:
    SPORTRADAR_API_KEY = SPORTRADAR_API_KEYS[0]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
EDGE_MIN = float(os.getenv("EDGE_MIN", "0.05"))
MIN_PROJECTED_MINUTES = float(os.getenv("MIN_PROJECTED_MINUTES", "15.0"))
# Skip player props when fewer than this many game logs exist (current season
# plus the prior-season blend). Only bites on rookies / players with no prior
# NBA season, whose projections would otherwise rest on 1-2 games.
MIN_PLAYER_GAMES = int(os.getenv("MIN_PLAYER_GAMES", "3"))
# Early-season team stats are shrunk toward last season's: current-season
# weight = GP / (GP + TEAM_PRIOR_GAMES). 15 → 1 GP is 6% current, 15 GP 50%.
TEAM_PRIOR_GAMES = float(os.getenv("TEAM_PRIOR_GAMES", "15"))
# Game-market alerts (moneyline / spread / totals / Q1 / H1 / team totals).
# Off by default: the team-level model is overconfident against sharp lines
# (every market on every game flagged at 5-50% "edges" in Oct 2026).
GAME_MARKETS_ENABLED = os.getenv("GAME_MARKETS_ENABLED", "false").lower() in ("true", "1", "yes")
# Props alert only once their game is within this many hours of tip-off, so
# the day's budget isn't spent on the first morning scan before lineup and
# injury news. Keep it above SCAN_INTERVAL_MINUTES so every game gets a scan
# inside the window.
ALERT_LEAD_HOURS = float(os.getenv("ALERT_LEAD_HOURS", "3"))
# Overs went 21-38 (hit 36% vs 51% implied) through 2026-10-26 while unders
# beat the market; off until the model's overs prove out.
OVERS_ENABLED = os.getenv("OVERS_ENABLED", "false").lower() in ("true", "1", "yes")
# While overs are off, paper-track the overs the raw model would bet
# (shadow_picks table; `python main.py shadow_stats`). No alerts, no stakes.
SHADOW_OVERS_ENABLED = os.getenv("SHADOW_OVERS_ENABLED", "true").lower() in ("true", "1", "yes")
# Parlay / SGP / slate-parlay alerts. Off: with k=0 calibration the legs have
# no edge beyond the market, so the "edges" multiply book hold; the same
# parlay was also re-sent every scan and none were graded.
PARLAYS_ENABLED = os.getenv("PARLAYS_ENABLED", "false").lower() in ("true", "1", "yes")
# Store every scanned prop side (scan_candidates) to score the model against
# the market on all props, not just alerts (`python main.py model_eval`).
CANDIDATE_LOG_ENABLED = os.getenv("CANDIDATE_LOG_ENABLED", "true").lower() in ("true", "1", "yes")
# Windows: keep the PC from idle-sleeping while the scheduler runs (src/utils/keep_awake.py).
KEEP_AWAKE = os.getenv("KEEP_AWAKE", "true").lower() in ("true", "1", "yes")
# Prop stake sizing: "flat" stakes BANKROLL * FLAT_STAKE_PCT on every play;
# "kelly" scales by estimated edge. Flat while edge size shows no relation to
# hit rate (13%+ edges hit exactly at implied through 2026-10-26).
STAKE_MODE = os.getenv("STAKE_MODE", "flat").lower()
FLAT_STAKE_PCT = float(os.getenv("FLAT_STAKE_PCT", "0.01"))

# -- Odds API credit budget (free tier: 500 credits/key/month) --------------
# Each event-odds request costs one credit per market returned (<=10 books is
# one billing group; /events is free). Defaults fit ~3,000 credits/month.
# Hard cap on credits spent per ET day across every caller (persisted in
# src/data/odds_spend.json, so restarts don't reset it).
ODDS_DAILY_CREDIT_BUDGET = int(os.getenv("ODDS_DAILY_CREDIT_BUDGET", "100"))
# Separate daily allowance for CLV closing-line fetches (1 credit per alerted
# market per game), so closing lines never starve the scans and vice versa.
ODDS_CLV_CREDIT_BUDGET = int(os.getenv("ODDS_CLV_CREDIT_BUDGET", "30"))
# Event odds are reused for this long, so every scan, news-triggered scan and
# CLV check within a game's window shares a single fetch.
ODDS_CACHE_MIN = float(os.getenv("ODDS_CACHE_MIN", "240"))
# The scan's sharp fetch happens once a game is this close to tip. Defaults to
# the alert window so the first scan allowed to alert has the sharp line.
SHARP_FETCH_HOURS = float(os.getenv("SHARP_FETCH_HOURS", str(ALERT_LEAD_HOURS)))
# Alternate-line markets double the sharp fetch (7 -> 14 markets).
ODDS_FETCH_ALTS = os.getenv("ODDS_FETCH_ALTS", "false").lower() in ("true", "1", "yes")
# scout_lines: 7 prop markets per game every BDL_SHARP_SCAN_INTERVAL minutes
# for steam snapshots (~23% of credit demand). BDL lines still feed
# line_history on every scan.
SCOUT_LINES_ENABLED = os.getenv("SCOUT_LINES_ENABLED", "false").lower() in ("true", "1", "yes")
# game_arb: Kalshi game-winner arb, 1 credit per game per GAME_ARB_ODDS_TTL_MIN.
GAME_ARB_ENABLED = os.getenv("GAME_ARB_ENABLED", "false").lower() in ("true", "1", "yes")

# Playoff overrides. Lines tighten (sharper books, more handle) and rotations
# compress (bench rarely sees the floor). When scan_props runs with
# PLAYOFF_MODE=True it selects these in place of the regular-season values.
PLAYOFF_EDGE_MIN              = float(os.getenv("PLAYOFF_EDGE_MIN", "0.06"))
PLAYOFF_MIN_PROJECTED_MINUTES = float(os.getenv("PLAYOFF_MIN_PROJECTED_MINUTES", "18.0"))

# Elimination-game guard. When every remaining game can end the series
# (e.g. the Finals at 3-1), markets are the sharpest of the year and
# blowout / garbage-time risk skews player props in both directions —
# starters get pulled early whichever way the game breaks. When
# ELIMINATION_MODE=true:
#   - every edge floor is raised by ELIMINATION_EDGE_BUMP (additive)
#   - every Kelly stake is multiplied by ELIMINATION_KELLY_MULT
# Set it for the remainder of the series; clear it when the season ends.
ELIMINATION_MODE       = os.getenv("ELIMINATION_MODE", "false").lower() in ("true", "1", "yes")
ELIMINATION_EDGE_BUMP  = float(os.getenv("ELIMINATION_EDGE_BUMP", "0.03"))
ELIMINATION_KELLY_MULT = float(os.getenv("ELIMINATION_KELLY_MULT", "0.5"))

# Per-market edge-floor overrides (applied on top of the dynamic floor).
# Floors are data-driven from settled bet history (negative ROI → raised floor).
# Blocks / Assists / PRA were already raised from prior analysis.
# Points raised to 0.08 due to persistent -3% ROI on marginal edges (< 8%).
# Threes raised to 0.07 due to high variance and recent underperformance.
PER_MARKET_EDGE_MIN: dict = {
    'player_points':                  float(os.getenv("EDGE_MIN_POINTS",   "0.08")),
    'player_threes':                  float(os.getenv("EDGE_MIN_THREES",   "0.07")),
    'player_blocks':                  float(os.getenv("EDGE_MIN_BLOCKS",   "0.09")),
    'player_assists':                 float(os.getenv("EDGE_MIN_ASSISTS",  "0.10")),
    'player_points_rebounds_assists': float(os.getenv("EDGE_MIN_PRA",     "0.12")),
}
ODDS_REGION = os.getenv("ODDS_REGION", "us")
BOOKMAKERS_RAW = os.getenv("BOOKMAKERS", "draftkings,fanduel,betmgm,caesars,betonlineag,lowvig")
BOOKMAKERS = BOOKMAKERS_RAW.split(",") if BOOKMAKERS_RAW else []

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "props.db")

# Run Risk Management
BANKROLL = float(os.getenv("BANKROLL", "1000.0"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
# Risk limits as fractions of BANKROLL. Daily and per-game totals are summed
# over alerts in the last 12 hours.
DAILY_RISK_PCT = float(os.getenv("DAILY_RISK_PCT", "0.25"))
GAME_RISK_PCT  = float(os.getenv("GAME_RISK_PCT",  "0.10"))
MAX_BET_PCT    = float(os.getenv("MAX_BET_PCT",    "0.025"))

# Scheduler credit-conservation settings
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "45"))
QUOTA_FLOOR = int(os.getenv("QUOTA_FLOOR", "30"))

# Off-season hibernation.
# "auto" (default): hibernate after OFF_SEASON_AFTER_SYNCS consecutive daily
# syncs return zero upcoming NBA events; wake automatically when events return.
# "true" / "false": force hibernation on or off regardless of the schedule.
OFF_SEASON             = os.getenv("OFF_SEASON", "auto").strip().lower()
OFF_SEASON_AFTER_SYNCS = int(os.getenv("OFF_SEASON_AFTER_SYNCS", "3"))
NEWS_POLL_INTERVAL     = int(os.getenv("NEWS_POLL_INTERVAL",     "60"))
TWITTER_POLL_INTERVAL  = int(os.getenv("TWITTER_POLL_INTERVAL",  "15"))

# Sharp-line devigging — top-down signal
SHARP_BOOKS_RAW = os.getenv("SHARP_BOOKS", "pinnacle,bookmaker,circa")
SHARP_BOOKS     = [b.strip() for b in SHARP_BOOKS_RAW.split(",") if b.strip()]
REC_BOOKS_RAW   = os.getenv("REC_BOOKS", "draftkings,fanduel")
REC_BOOKS       = [b.strip() for b in REC_BOOKS_RAW.split(",") if b.strip()]
SHARP_EDGE_MIN  = float(os.getenv("SHARP_EDGE_MIN", "0.03"))
PLAYOFF_SHARP_EDGE_MIN = float(os.getenv("PLAYOFF_SHARP_EDGE_MIN", "0.035"))

# Consensus sharp books for synthetic true probability
CONSENSUS_BOOKS_RAW = os.getenv("CONSENSUS_BOOKS", "pinnacle,circa,bookmaker")
CONSENSUS_BOOKS     = [b.strip().lower() for b in CONSENSUS_BOOKS_RAW.split(",") if b.strip()]
CONSENSUS_HOLD_MAX  = float(os.getenv("CONSENSUS_HOLD_MAX", "0.06"))

# ── BallDontLie GOAT Tier ─────────────────────────────────────────────
BDL_API_KEY = os.getenv("BDL_API_KEY", "")
# Use BDL as primary data source for props, injuries, lineups
# Falls back to Odds API + scraping when BDL_API_KEY is empty
BDL_ENABLED = bool(BDL_API_KEY)
# BDL season profiles (usage%, touches, playtypes, drives) for XGBoost. Off:
# the models are trained without them (zeros / box-score proxies), and the
# profile parser never matched BDL's nested player.id / stats rows, so live
# features already equal the training defaults. Each scan spent ~10 BDL
# calls per game on them, feeding the rate limiting. Enable only together
# with a parser fix and a retrain on the same features.
BDL_PROFILES_ENABLED = os.getenv("BDL_PROFILES_ENABLED", "false").lower() in ("true", "1", "yes")
# BDL vendors to pull prop lines from (all go to line_history for market
# data). Alerts are priced only at REC_BOOKS, the books the user can bet.
# Sharp pricing still from Odds API.
BDL_PROP_VENDORS_RAW = os.getenv("BDL_PROP_VENDORS", "draftkings,fanduel,caesars,betmgm")
BDL_PROP_VENDORS = [v.strip() for v in BDL_PROP_VENDORS_RAW.split(",") if v.strip()]
# When BDL is active, reduce Odds API scan frequency (only needed for Pinnacle)
BDL_SHARP_SCAN_INTERVAL = int(os.getenv("BDL_SHARP_SCAN_INTERVAL", "30"))  # minutes

PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_blocks",
    "player_steals",
]

# Alternate-line markets (fat-tail pricing inefficiency).
# Books underprice extreme alt-lines; the same distribution engine
# produces sharper probabilities at these tails than the book's flat markup.
ALT_PROP_MARKETS = [
    "player_points_alternate",
    "player_rebounds_alternate",
    "player_assists_alternate",
    "player_threes_alternate",
    "player_points_rebounds_assists_alternate",
    "player_blocks_alternate",
    "player_steals_alternate",
]

# Maps each alt-market back to its statistical base market for projection/distribution.
ALT_TO_BASE_MARKET: dict = {
    "player_points_alternate":                  "player_points",
    "player_rebounds_alternate":                "player_rebounds",
    "player_assists_alternate":                 "player_assists",
    "player_threes_alternate":                  "player_threes",
    "player_points_rebounds_assists_alternate": "player_points_rebounds_assists",
    "player_blocks_alternate":                  "player_blocks",
    "player_steals_alternate":                  "player_steals",
}

# ── Quant Edge Copilot (OpenRouter LLM narration layer) ───────────────────────
# The copilot is a NARRATION-ONLY assistant: it explains picks the quant stack
# (XGBoost + calibration + edge_ranker) has ALREADY produced. It never computes
# edges, never invents statistics, and never makes betting decisions. All
# numbers it cites come verbatim from alerts_sent rows we hand it.
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
COPILOT_ENABLED     = bool(OPENROUTER_API_KEY) and \
    os.getenv("COPILOT_ENABLED", "true").lower() in ("true", "1", "yes")

# Model slugs MUST be valid OpenRouter ids — verify at https://openrouter.ai/models.
# Narration is a cheap/fast task, so default to a low-cost model; fall back to a
# stronger model only if the primary errors. Defaults reuse slugs already proven
# in the sibling unified-sports-bot client.
COPILOT_MODEL = os.getenv("COPILOT_MODEL", "google/gemini-2.5-flash")
COPILOT_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "COPILOT_FALLBACK_MODELS", "anthropic/claude-sonnet-4,deepseek/deepseek-r1"
    ).split(",") if m.strip()
]

# Hard daily spend cap (USD, approximate) before the copilot stops answering.
COPILOT_DAILY_USD_LIMIT = float(os.getenv("COPILOT_DAILY_USD_LIMIT", "2.0"))

# Per-chat throttle: at most COPILOT_RATE_MAX questions per rolling window.
COPILOT_RATE_MAX      = int(os.getenv("COPILOT_RATE_MAX", "5"))
COPILOT_RATE_WINDOW_S = int(os.getenv("COPILOT_RATE_WINDOW_S", "60"))

# Grounding context size: how many recent picks to feed the copilot, how far back.
COPILOT_CONTEXT_ALERTS = int(os.getenv("COPILOT_CONTEXT_ALERTS", "20"))
COPILOT_CONTEXT_HOURS  = int(os.getenv("COPILOT_CONTEXT_HOURS", "36"))
