"""
Multi-season XGBoost training pipeline.

Fetches game logs for all active players across the in-progress season
plus the last N completed seasons,
builds the full training dataset, and trains one XGBRegressor per market.
Existing models are overwritten on success.

Usage:
    python -m src.pipelines.train_ml
    TRAIN_SEASONS=2 python -m src.pipelines.train_ml

Environment:
    TRAIN_SEASONS   number of completed seasons to use alongside the
                    current one (default 3)
    TRAIN_MIN_GP    minimum games played filter (default 20)
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.utils.logging_utils import get_logger
from src.clients.nba_stats import (
    NbaStatsClient, get_current_nba_season, get_completed_seasons, get_current_roster,
    _normalize_player_name,
)
from src.clients.bdl_client import BDLClient
from src.clients.bdl_game_logs import BDLGameLogs
from src.data.db import get_db_client
from src.clients.telegram_bot import TelegramBotClient
from src.models.ml_model import train_models_from_logs, train_models_with_clv_feedback
from src.models.experiment_tracker import training_run

logger = get_logger(__name__)

_N_SEASONS       = int(os.getenv('TRAIN_SEASONS', '3'))
_MIN_GP          = int(os.getenv('TRAIN_MIN_GP',  '20'))


def _default_seasons() -> List[str]:
    """In-progress season first, then the last _N_SEASONS completed ones."""
    return [get_current_nba_season()] + get_completed_seasons(_N_SEASONS)


def _nba_to_bdl_ids(bdl_client: BDLClient) -> Dict[int, Tuple[str, int]]:
    """
    Map NBA Stats player ID → (full name, BDL player ID). The two APIs use
    different ID spaces, so they are bridged on normalized player name.
    """
    bdl_by_name: Dict[str, int] = {}
    bdl_by_last: Dict[str, List[int]] = {}
    for p in bdl_client.get_active_players():
        norm = _normalize_player_name(f"{p['first_name']} {p['last_name']}")
        bdl_by_name[norm] = p['id']
        if norm:
            bdl_by_last.setdefault(norm.split()[-1], []).append(p['id'])

    from nba_api.stats.static import players as nba_players
    nba_id_to_name = {p['id']: p['full_name'] for p in nba_players.get_players()}
    # Rookies / new signings aren't in the bundled static list
    roster = get_current_roster()['id_to_name']
    nba_id_to_name.update(roster)

    mapping: Dict[int, Tuple[str, int]] = {}
    claimed: set = set()
    # Current-roster players first, so a retired namesake (suffixes are
    # normalized away: "Tim Hardaway" / "Tim Hardaway Jr.") never takes an
    # active player's BDL ID.
    ordered = sorted(nba_id_to_name.items(), key=lambda kv: kv[0] not in roster)
    for nba_id, name in ordered:
        norm = _normalize_player_name(name)
        if not norm:
            continue
        # Exact, then reversed order ("Yang Hansen" / "Hansen Yang")
        bdl_id = bdl_by_name.get(norm) or bdl_by_name.get(' '.join(reversed(norm.split())))
        # Nicknames ("Bones Hyland" / "Nah'Shon Hyland"): a unique last name
        # is safe, but only for players on a current roster so retired
        # namesakes can't steal an active player's ID.
        if not bdl_id and nba_id in roster:
            candidates = bdl_by_last.get(norm.split()[-1], [])
            if len(candidates) == 1:
                bdl_id = candidates[0]
        if bdl_id and bdl_id not in claimed:
            mapping[nba_id] = (name, bdl_id)
            claimed.add(bdl_id)
    return mapping


def train_ml_models(seasons: Optional[List[str]] = None) -> dict:
    """
    Fetch multi-season logs for all active players and retrain XGBoost models.
    Returns {market: success_bool}.
    """
    if seasons is None:
        seasons = _default_seasons()

    stats = NbaStatsClient()
    bot   = TelegramBotClient()
    bdl_logs = BDLGameLogs(BDLClient(), db=get_db_client())

    logger.info(f"ML training started: seasons={seasons}")
    bot.send_message(
        f"🤖 <b>ML Training started</b>\nSeasons: {', '.join(seasons)}\n"
        f"Fetching player list..."
    )

    player_ids = stats.get_all_active_player_ids(min_gp=_MIN_GP)
    logger.info(f"Training on {len(player_ids)} players across {len(seasons)} seasons.")

    # Fetch matchup context once (current-season team stats for feature lookups)
    try:
        opp_stats_df  = stats.get_opponent_stats()
        team_stats_df = stats.get_team_stats()
        def_stats_df  = stats.get_team_defense_stats()
    except Exception as e:
        logger.warning(f"Could not fetch team stats for training context: {e}")
        opp_stats_df = team_stats_df = def_stats_df = None

    # Compute league-average pace for normalization
    league_avg_pace = 99.0
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        league_avg_pace = float(team_stats_df['PACE'].mean())

    logger.info("Fetching BDL player mapping...")
    id_map = _nba_to_bdl_ids(BDLClient())

    # Collect all logs
    player_logs: List[pd.DataFrame] = []
    for idx, pid in enumerate(player_ids):
        if pid not in id_map:
            logger.warning(f"Could not find BDL ID for NBA player {pid}. Skipping.")
            continue
        _, bdl_id = id_map[pid]

        season_dfs = []
        for season in seasons:
            szn_int = int(season.split('-')[0])
            df = bdl_logs.get_player_game_logs(
                bdl_id, szn_int, ignore_ttl=(season != stats.season))
            if not df.empty:
                season_dfs.append(df)

        if not season_dfs:
            continue

        combined = pd.concat(season_dfs, ignore_index=True)
        if not combined.empty and 'MIN' in combined.columns:
            # Sort newest-first (nba_api convention expected by build_training_data)
            if 'GAME_DATE' in combined.columns:
                try:
                    combined['_dt'] = pd.to_datetime(combined['GAME_DATE'], errors='coerce')
                    combined = combined.sort_values('_dt', ascending=False).drop(columns=['_dt'])
                except Exception:
                    pass
            player_logs.append(combined)

        if (idx + 1) % 50 == 0:
            logger.info(f"Fetched logs: {idx + 1}/{len(player_ids)} players")

    logger.info(f"Collected logs for {len(player_logs)} players. Training XGBoost...")

    run_name = f"train_ml_{datetime.now().strftime('%Y%m%d_%H%M')}"
    with training_run(run_name, tags={"type": "xgboost_weekly", "seasons": ",".join(seasons)}) as run:
        run.log_params({
            "seasons": ",".join(seasons),
            "n_seasons": len(seasons),
            "min_gp": _MIN_GP,
            "n_players": len(player_logs),
        })

        results = train_models_from_logs(
            player_logs_list=player_logs,
            opp_stats_df=opp_stats_df,
            team_stats_df=team_stats_df,
            def_stats_df=def_stats_df,
            league_avg_pace=league_avg_pace,
        )

        for market, ok in results.items():
            run.log_metric(f"trained_{market}", int(ok))

        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb')
        for market, ok in results.items():
            if ok:
                artifact = os.path.join(model_dir, f"player_{market}_ensemble.pkl")
                run.log_artifact(artifact)

    trained  = [m for m, ok in results.items() if ok]
    skipped  = [m for m, ok in results.items() if not ok]
    msg = (
        f"✅ <b>ML Training Complete</b>\n\n"
        f"Trained: {', '.join(trained) or 'none'}\n"
        f"Skipped (insufficient data): {', '.join(skipped) or 'none'}\n"
        f"Players: {len(player_logs)} | Seasons: {', '.join(seasons)}"
    )
    bot.send_message(msg)
    logger.info(f"ML training results: {results}")
    return results


def train_ml_models_clv_feedback(seasons: Optional[List[str]] = None) -> dict:
    """
    Monthly CLV-feedback retraining.

    Identical log-collection pipeline as ``train_ml_models``, but passes
    (player_name, logs) tuples to ``train_models_with_clv_feedback`` so
    XGBoost sample weights are derived from historical CLV outcomes.

    Scheduled once a month (first Sunday).  Falls back to uniform weights
    when the clv_tracking table has insufficient data.
    """
    if seasons is None:
        seasons = _default_seasons()

    stats = NbaStatsClient()
    bot   = TelegramBotClient()
    db    = get_db_client()
    bdl_logs = BDLGameLogs(BDLClient(), db=db)

    logger.info(f"CLV feedback training: seasons={seasons}")
    bot.send_message(
        f"🧠 <b>CLV Feedback Training started</b>\nSeasons: {', '.join(seasons)}\n"
        f"Fetching player list..."
    )

    player_ids = stats.get_all_active_player_ids(min_gp=_MIN_GP)

    try:
        opp_stats_df  = stats.get_opponent_stats()
        team_stats_df = stats.get_team_stats()
        def_stats_df  = stats.get_team_defense_stats()
    except Exception as e:
        logger.warning(f"CLV training: could not fetch team stats: {e}")
        opp_stats_df = team_stats_df = def_stats_df = None

    league_avg_pace = 99.0
    if team_stats_df is not None and not team_stats_df.empty and 'PACE' in team_stats_df.columns:
        league_avg_pace = float(team_stats_df['PACE'].mean())

    id_map = _nba_to_bdl_ids(BDLClient())

    # Collect (player_name, logs) — name required for CLV lookup
    named_logs: list = []
    for idx, pid in enumerate(player_ids):
        if pid not in id_map:
            logger.warning(f"Could not find BDL ID for NBA player {pid}. Skipping.")
            continue
        player_name, bdl_id = id_map[pid]

        season_dfs = []
        for season in seasons:
            szn_int = int(season.split('-')[0])
            df = bdl_logs.get_player_game_logs(
                bdl_id, szn_int, ignore_ttl=(season != stats.season))
            if not df.empty:
                season_dfs.append(df)

        if not season_dfs:
            continue

        combined = pd.concat(season_dfs, ignore_index=True)
        if not combined.empty and 'MIN' in combined.columns:
            if 'GAME_DATE' in combined.columns:
                try:
                    combined['_dt'] = pd.to_datetime(combined['GAME_DATE'], errors='coerce')
                    combined = combined.sort_values('_dt', ascending=False).drop(columns=['_dt'])
                except Exception:
                    pass
            named_logs.append((player_name, combined))

        if (idx + 1) % 50 == 0:
            logger.info(f"CLV training logs: {idx + 1}/{len(player_ids)} players")

    logger.info(f"CLV training: {len(named_logs)} players. Running weighted fit...")

    results = train_models_with_clv_feedback(
        player_named_logs=named_logs,
        db=db,
        opp_stats_df=opp_stats_df,
        team_stats_df=team_stats_df,
        def_stats_df=def_stats_df,
        league_avg_pace=league_avg_pace,
    )

    trained = [m for m, ok in results.items() if ok]
    skipped = [m for m, ok in results.items() if not ok]
    bot.send_message(
        f"✅ <b>CLV Feedback Training Complete</b>\n\n"
        f"Trained: {', '.join(trained) or 'none'}\n"
        f"Skipped: {', '.join(skipped) or 'none'}\n"
        f"Players: {len(named_logs)} | Seasons: {', '.join(seasons)}"
    )
    return results


if __name__ == '__main__':
    train_ml_models()
