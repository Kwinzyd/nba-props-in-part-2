"""Past-only box-score rate models and chronological candidate training.

Schema version 3 retains reconstructible rolling, schedule and foul features.
Unarchived matchup/tracking inputs are excluded from the fitted vector.
Training stages candidates; prospective price-aware promotion is separate.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from src.clients.travel_fatigue import travel_features_for_game
from src.models.distributions import compute_player_foul_rate

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb')
MIN_TRAIN_SAMPLES = 50
FEATURE_SCHEMA_VERSION = 3  # canonical game type and shared personal-foul features

_MARKET_COL = {
    'player_points':   'PTS',
    'player_rebounds': 'REB',
    'player_assists':  'AST',
    'player_threes':   'FG3M',
    'player_blocks':   'BLK',
    'player_steals':   'STL',
}

FEATURE_NAMES = [
    # Base rates (expanded with 20-game windows)
    'rate_5g', 'rate_10g', 'rate_20g', 'rate_season',
    'ewm_rate_5g', 'ewm_rate_10g', 'ewm_rate_20g',
    # Minutes
    'avg_min_5g', 'std_min_5g', 'max_min_10g',
    # Situational
    'home_flag', 'rest_days', 'b2b_flag', 'playoff_flag', 'playoff_share_5g',
    'games_played',
    'streak_factor', 'home_rate_delta', 'series_game_num',
    # Usage & efficiency (BDL real at inference; proxy/computed at training)
    'real_usage_pct', 'ts_pct_5g',
    # Travel fatigue
    'travel_miles', 'tz_shift_hours', 'altitude_flag',
    # Matchup context
    'opp_def_rating', 'pace_factor',
    'opp_pace', 'opp_rebound_pct', 'opp_pts_paint',
    # BDL season profile: tracking + playtype (real at inference, 0.0 at training)
    'avg_touches', 'pnr_bh_freq', 'pnr_roll_freq', 'iso_freq', 'spotup_freq',
    'transition_freq', 'postup_freq', 'drives_per_game',
    # BDL V2 advanced tracking (real at inference, 0.0 at training)
    'avg_speed', 'avg_contested_fg_pct', 'avg_deflections',
    'avg_points_paint', 'avg_pct_pts_paint',
    # Foul rate: per-minute foul rate from recent logs (minutes variance signal)
    'player_foul_rate',
]

# Keep only features reconstructible from pregame historical box scores or schedule.
# Live tracking profiles and current-season matchup tables have no as-of archive.
_UNARCHIVED = {'opp_def_rating', 'pace_factor', 'opp_pace', 'opp_rebound_pct',
              'opp_pts_paint', 'avg_touches', 'pnr_bh_freq', 'pnr_roll_freq',
              'iso_freq', 'spotup_freq', 'transition_freq', 'postup_freq',
              'drives_per_game', 'avg_speed', 'avg_contested_fg_pct',
              'avg_deflections', 'avg_points_paint', 'avg_pct_pts_paint'}
FEATURE_NAMES = [name for name in FEATURE_NAMES if name not in _UNARCHIVED]

# Opponent stat column per market (mirrors nba_stats._MARKET_OPP_COL)
_MARKET_OPP_STAT = {
    'player_points':   'OPP_PTS',
    'player_rebounds': 'OPP_REB',
    'player_assists':  'OPP_AST',
    'player_threes':   'OPP_FG3M',
}


def _extract_opp_abbr(matchup: str) -> str:
    """Extract opponent team abbreviation from a MATCHUP string like 'LAL vs. GSW'."""
    matchup = matchup.strip()
    if 'vs.' in matchup:
        return matchup.split('vs.')[-1].strip().upper()
    if '@' in matchup:
        return matchup.split('@')[-1].strip().upper()
    return ''


def _compute_pace_factor(matchup: str, opp_abbr: str,
                          pace_lookup: Optional[Dict[str, float]],
                          league_avg_pace: float) -> float:
    """Compute (team_pace + opp_pace) / (2 * league_avg). Falls back to 1.0."""
    if not pace_lookup or league_avg_pace <= 0:
        return 1.0
    own_abbr = matchup.split()[0].upper() if matchup else ''
    own_pace = pace_lookup.get(own_abbr, league_avg_pace)
    opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
    return (own_pace + opp_pace) / (2.0 * league_avg_pace)


class PropMLModel:
    """XGBoost model for a single prop market (base learner in ensemble)."""

    def __init__(self, market: str):
        self.market = market
        self.col = _MARKET_COL.get(market, 'PTS')
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.model_path = os.path.join(MODEL_DIR, f'{market}_xgb.pkl')
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                if getattr(self.model, 'prop_feature_schema_version', None) != FEATURE_SCHEMA_VERSION:
                    self.model = None
            except Exception:
                self.model = None

    def _safe_rate(self, df_slice: pd.DataFrame) -> float:
        if df_slice.empty or 'MIN' not in df_slice.columns:
            return 0.0
        total_min = df_slice['MIN'].sum()
        if total_min <= 0:
            return 0.0
        return float(df_slice[self.col].sum() / total_min) if self.col in df_slice.columns else 0.0

    def _ewm_rate(self, logs: pd.DataFrame, n: int = 5, alpha: float = 0.3) -> float:
        """Exponentially-weighted per-minute rate over the last n games (newest first)."""
        recent = logs.head(n)
        if recent.empty or self.col not in recent.columns or 'MIN' not in recent.columns:
            return 0.0
        rates = []
        for _, row in recent.iterrows():
            m = float(row.get('MIN', 0) or 0)
            s = float(row.get(self.col, 0) or 0)
            if m > 0:
                rates.append(s / m)
        if not rates:
            return 0.0
        return float(pd.Series(rates[::-1]).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

    def _usage_proxy(self, logs: pd.DataFrame, n: int = 5) -> float:
        """(FGA + 0.44*FTA + TOV) / MIN averaged over last n games."""
        recent = logs.head(n)
        if recent.empty or 'MIN' not in recent.columns:
            return 0.0
        total_min = recent['MIN'].sum()
        if total_min <= 0:
            return 0.0
        fga = recent.get('FGA', pd.Series([0] * len(recent))).sum()
        fta = recent.get('FTA', pd.Series([0] * len(recent))).sum()
        tov = recent.get('TOV', pd.Series([0] * len(recent))).sum()
        return float((fga + 0.44 * fta + tov) / total_min)

    def _ts_pct(self, logs: pd.DataFrame, n: int = 5) -> float:
        """True shooting % = PTS / (2 * (FGA + 0.44 * FTA)) over last n games."""
        recent = logs.head(n)
        if recent.empty:
            return 0.0
        pts = recent.get('PTS', pd.Series([0] * len(recent))).sum()
        fga = recent.get('FGA', pd.Series([0] * len(recent))).sum()
        fta = recent.get('FTA', pd.Series([0] * len(recent))).sum()
        denom = 2.0 * (fga + 0.44 * fta)
        return float(pts / denom) if denom > 0 else 0.0

    def _streak_factor(self, logs: pd.DataFrame) -> float:
        """Rate last 3 games / rate games 4-10. Clamped [0.5, 2.0]."""
        recent_3 = self._safe_rate(logs.head(3))
        prev_7   = self._safe_rate(logs.iloc[3:10])
        if prev_7 <= 0:
            return 1.0
        return float(max(0.5, min(2.0, recent_3 / prev_7)))

    def _home_rate_delta(self, logs: pd.DataFrame) -> float:
        """Historical per-minute rate in home games minus away games."""
        if 'MATCHUP' not in logs.columns:
            return 0.0
        home_logs = logs[logs['MATCHUP'].str.contains('vs\\.', na=False)].head(30)
        away_logs = logs[logs['MATCHUP'].str.contains('@', na=False)].head(30)
        return self._safe_rate(home_logs) - self._safe_rate(away_logs)

    def _playoff_share(self, logs: pd.DataFrame, n: int = 5) -> float:
        """
        Fraction of the last n games that were playoff games (SEASON_ID starts '4').
        Signals how much of the rolling-rate window is playoff-composed so the
        model can learn the RS→PO drift implicit in rate_5g / ewm_rate_5g.
        """
        recent = logs.head(n)
        if recent.empty or 'SEASON_ID' not in recent.columns:
            return 0.0
        is_po = recent['SEASON_ID'].astype(str).str.startswith('4')
        return float(is_po.sum()) / float(len(recent))

    def build_features(self, logs: pd.DataFrame,
                       home_flag: bool = False,
                       rest_days: int = 2,
                       playoff_flag: bool = False,
                       opp_abbr: str = "",
                       opp_def_rating: float = 1.0,
                       pace_factor: float = 1.0,
                       opp_pace: float = 1.0,
                       opp_rebound_pct: float = 1.0,
                       opp_pts_paint: float = 1.0,
                       travel_miles: float = 0.0,
                       tz_shift_hours: int = 0,
                       altitude_flag: bool = False,
                       # BDL season profile (real at inference; computed/0 at training)
                       real_usage_pct: float = 0.0,
                       avg_touches: float = 0.0,
                       pnr_bh_freq: float = 0.0,
                       pnr_roll_freq: float = 0.0,
                       iso_freq: float = 0.0,
                       spotup_freq: float = 0.0,
                       transition_freq: float = 0.0,
                       postup_freq: float = 0.0,
                       drives_per_game: float = 0.0,
                       ts_pct: float = 0.0,
                       # BDL V2 advanced tracking (real at inference; 0 at training)
                       avg_speed: float = 0.0,
                       avg_contested_fg_pct: float = 0.0,
                       avg_deflections: float = 0.0,
                       avg_points_paint: float = 0.0,
                       avg_pct_pts_paint: float = 0.0,
                       player_foul_rate: Optional[float] = None) -> Optional[Dict]:
        """Build feature dict from game logs. Returns None if insufficient data."""
        if logs.empty or len(logs) < 3:
            return None
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return None

        r5  = logs.head(5)
        r10 = logs.head(10)
        r20 = logs.head(20)
        avg_min = r5['MIN'].mean()
        std_min = r5['MIN'].std() if len(r5) > 1 else 0.0

        # Calculate series game number (consecutive games vs same opponent)
        series_count = 1
        if opp_abbr and 'MATCHUP' in logs.columns:
            for _, row in logs.iterrows():
                hist_opp = _extract_opp_abbr(str(row.get('MATCHUP', '')))
                if hist_opp == opp_abbr:
                    series_count += 1
                else:
                    break

        return {
            # Base rates (expanded)
            'rate_5g':          self._safe_rate(r5),
            'rate_10g':         self._safe_rate(r10),
            'rate_20g':         self._safe_rate(r20),
            'rate_season':      self._safe_rate(logs),
            'ewm_rate_5g':      self._ewm_rate(logs, n=5, alpha=0.3),
            'ewm_rate_10g':     self._ewm_rate(logs, n=10, alpha=0.25),
            'ewm_rate_20g':     self._ewm_rate(logs, n=20, alpha=0.15),
            # Minutes
            'avg_min_5g':       float(avg_min) if not pd.isna(avg_min) else 0.0,
            'std_min_5g':       float(std_min) if not pd.isna(std_min) else 0.0,
            'max_min_10g':      float(r10['MIN'].max()) if not r10.empty else 0.0,
            # Situational
            'home_flag':        int(home_flag),
            'rest_days':        min(rest_days, 7),
            'b2b_flag':         int(rest_days == 0),
            'playoff_flag':     int(playoff_flag),
            'playoff_share_5g': self._playoff_share(logs, n=5),
            'games_played':     len(logs),
            'streak_factor':    self._streak_factor(logs),
            'home_rate_delta':  self._home_rate_delta(logs),
            'series_game_num':  series_count,
            # Usage & efficiency
            'real_usage_pct':   self._usage_proxy(logs, n=5),
            'ts_pct_5g':        self._ts_pct(logs, n=5),
            # Travel fatigue
            'travel_miles':     float(travel_miles),
            'tz_shift_hours':   int(tz_shift_hours),
            'altitude_flag':    int(altitude_flag),
            # Matchup context
            'opp_def_rating':   float(opp_def_rating),
            'pace_factor':      float(pace_factor),
            'opp_pace':         float(opp_pace),
            'opp_rebound_pct':  float(opp_rebound_pct),
            'opp_pts_paint':    float(opp_pts_paint),
            # BDL season profile
            'avg_touches':           float(avg_touches),
            'pnr_bh_freq':           float(pnr_bh_freq),
            'pnr_roll_freq':         float(pnr_roll_freq),
            'iso_freq':              float(iso_freq),
            'spotup_freq':           float(spotup_freq),
            'transition_freq':       float(transition_freq),
            'postup_freq':           float(postup_freq),
            'drives_per_game':       float(drives_per_game),
            # BDL V2 advanced tracking
            'avg_speed':             float(avg_speed),
            'avg_contested_fg_pct':  float(avg_contested_fg_pct),
            'avg_deflections':       float(avg_deflections),
            'avg_points_paint':      float(avg_points_paint),
            'avg_pct_pts_paint':     float(avg_pct_pts_paint),
            # Foul rate
            'player_foul_rate':      compute_player_foul_rate(logs) if player_foul_rate is None else float(player_foul_rate),
        }

    def build_training_data(self, logs: pd.DataFrame,
                            opp_def_lookup: Optional[Dict[str, float]] = None,
                            pace_lookup: Optional[Dict[str, float]] = None,
                            league_avg_pace: float = 99.0,
                            opp_rebound_pct_lookup: Optional[Dict[str, float]] = None,
                            opp_pts_paint_lookup: Optional[Dict[str, float]] = None,
                            player_name: str = "",
                            clv_weight_lookup: Optional[Dict] = None,
                            return_weights: bool = False,
                            return_metadata: bool = False,
                            target_since: Optional[str] = None,
                            ) -> Tuple:
        """
        Generate (X, y) training samples from a player's chronological game logs.
        Requires at least 15 games. Only uses past games as features (no lookahead).
        """
        if logs.empty or len(logs) < 15:
            return tuple([] for _ in range(2 + int(return_weights) + int(return_metadata)))
        if self.col not in logs.columns or 'MIN' not in logs.columns:
            return tuple([] for _ in range(2 + int(return_weights) + int(return_metadata)))

        if 'GAME_DATE' not in logs:
            return tuple([] for _ in range(2 + int(return_weights) + int(return_metadata)))
        # Sort target timestamps explicitly; callers can supply player-order data.
        chron = logs.copy()
        chron['_date'] = pd.to_datetime(chron['GAME_DATE'], errors='coerce', utc=True)
        chron = chron.dropna(subset=['_date']).sort_values('_date').reset_index(drop=True)
        player_norm = _norm_player_name(player_name)

        X, y, weights, metadata = [], [], [], []
        for i in range(10, len(chron)):
            current = chron.iloc[i]
            if target_since and current['_date'].date().isoformat() < target_since:
                continue
            if float(current.get('MIN', 0) or 0) <= 0:
                continue

            target_rate = float(current[self.col]) / float(current['MIN'])
            history = chron.iloc[:i][chron.iloc[:i]['_date'] < current['_date']].iloc[::-1]
            if len(history) < 3:
                continue

            matchup = str(current.get('MATCHUP', ''))
            is_home = 'vs.' in matchup
            opp_abbr = _extract_opp_abbr(matchup)

            try:
                cur_date  = pd.to_datetime(current['GAME_DATE'])
                prev_date = pd.to_datetime(history.iloc[0]['GAME_DATE'])
                rest = max(0, int((cur_date - prev_date).days) - 1)
            except Exception:
                rest = 2

            opp_def = opp_def_lookup.get(opp_abbr, 1.0) if opp_def_lookup else 1.0
            pace    = _compute_pace_factor(matchup, opp_abbr,
                                           pace_lookup, league_avg_pace)

            if pace_lookup and league_avg_pace > 0:
                raw_opp_pace = pace_lookup.get(opp_abbr, league_avg_pace)
                opp_pace_norm = raw_opp_pace / league_avg_pace
            else:
                opp_pace_norm = 1.0

            opp_reb_pct = (opp_rebound_pct_lookup.get(opp_abbr, 1.0)
                           if opp_rebound_pct_lookup else 1.0)
            opp_paint   = (opp_pts_paint_lookup.get(opp_abbr, 1.0)
                           if opp_pts_paint_lookup else 1.0)

            own_abbr = str(current.get('TEAM_ABBREVIATION', '')).upper()
            prev_matchup = str(chron.iloc[i - 1].get('MATCHUP', ''))
            t_miles, t_tz, t_alt = travel_features_for_game(matchup, prev_matchup, own_abbr)

            is_playoff = str(current.get('SEASON_ID', '')).startswith('4')

            _hist_foul_rate = compute_player_foul_rate(history)

            feats = self.build_features(
                history, is_home, rest, is_playoff, opp_abbr, opp_def, pace,
                opp_pace_norm, opp_reb_pct, opp_paint,
                travel_miles=t_miles, tz_shift_hours=t_tz, altitude_flag=t_alt,
                player_foul_rate=_hist_foul_rate,
            )
            if feats is None:
                continue

            sample_weight = 1.0
            X.append([feats[k] for k in FEATURE_NAMES])
            y.append(target_rate)
            weights.append(sample_weight)
            day = current['_date'].date().isoformat()
            # A date groups every game; the event key preserves identity when
            # multiple player logs represent the same game.
            matchup_key = '|'.join(sorted(matchup.replace('vs.', ' ').replace('@', ' ').split()))
            event_id = str(current.get('GAME_ID', '') or '')
            metadata.append({'game_date': day, 'event_id': (str(logs.attrs.get('game_id_provider', 'nba')) + ':' + event_id) if event_id and event_id != 'nan' else day + ':' + matchup_key,
                             'baseline_rate': float(feats['rate_10g'])})

        result = [X, y]
        if return_weights:
            result.append(weights)
        if return_metadata:
            result.append(metadata)
        return tuple(result)

    def train(self, X: List, y: List, sample_weight: Optional[List] = None, metadata=None) -> bool:
        """Stage a dated candidate; active artifacts are never overwritten here."""
        from src.models.model_validation import train_rate_candidate
        from src.utils.logging_utils import get_logger
        logger = get_logger(__name__)
        try:
            self.last_candidate_manifest = str(train_rate_candidate(
                self, X, y, metadata, sample_weight, ensemble=False))
            logger.info('Candidate staged for prospective evaluation: %s', self.last_candidate_manifest)
            return True
        except Exception as exc:
            logger.error('Candidate training failed closed [%s]: %s', self.market, exc)
            return False

    def predict_rate(self, features: Dict) -> Optional[float]:
        """Predict per-minute stat rate. Returns None if no model trained yet."""
        if self.model is None:
            return None
        try:
            feat_vec = np.array([[features[k] for k in FEATURE_NAMES]])
            return float(self.model.predict(feat_vec)[0])
        except Exception:
            return None


# =========================================================================
# Ensemble model: XGBoost + RandomForest + Ridge stacking
# =========================================================================

class EnsembleModel:
    """
    Stacking ensemble for a single prop market.

    Base learners:
      1. XGBRegressor  (tree-based, handles interactions)
      2. RandomForest   (tree-based, reduces XGB overfitting)
      3. LinearRegression (linear, captures main effects XGB might overfit)

    Meta-learner: Ridge on a strictly later date partition than base fitting.
    """

    def __init__(self, market: str):
        self.market = market
        self.col = _MARKET_COL.get(market, 'PTS')
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.ensemble_path = os.path.join(MODEL_DIR, f'{market}_ensemble.pkl')
        self.ensemble = None
        self._load()

    def _load(self):
        if os.path.exists(self.ensemble_path):
            try:
                import joblib
                loaded = joblib.load(self.ensemble_path)
                stored_feats = loaded.get('feature_names', []) if isinstance(loaded, dict) else []
                if stored_feats != FEATURE_NAMES or loaded.get('feature_schema_version') != FEATURE_SCHEMA_VERSION:
                    # Feature set changed since training → discard rather than
                    # feed a wrong-shape vector. Scan will fall back to the
                    # Bayesian projection until train_ml is re-run.
                    self.ensemble = None
                else:
                    self.ensemble = loaded
            except Exception:
                self.ensemble = None

    def train(self, X: List, y: List, sample_weight: Optional[List] = None, metadata=None) -> bool:
        """Stage a dated candidate; active artifacts are never overwritten here."""
        from src.models.model_validation import train_rate_candidate
        from src.utils.logging_utils import get_logger
        logger = get_logger(__name__)
        try:
            self.last_candidate_manifest = str(train_rate_candidate(
                self, X, y, metadata, sample_weight, ensemble=True))
            logger.info('Candidate staged for prospective evaluation: %s', self.last_candidate_manifest)
            return True
        except Exception as exc:
            logger.error('Candidate training failed closed [%s]: %s', self.market, exc)
            return False

    def predict_rate(self, features: Dict) -> Optional[float]:
        """Predict per-minute rate using ensemble."""
        if self.ensemble is None:
            return None
        try:
            feat_vec = np.array([[features[k] for k in FEATURE_NAMES]])
            base_preds = []
            for name, model in self.ensemble['bases']:
                base_preds.append(model.predict(feat_vec)[0])
            meta_input = np.array([base_preds])
            return float(self.ensemble['meta'].predict(meta_input)[0])
        except Exception:
            return None


def _clone_model(model):
    """Clone a sklearn/xgb model by re-instantiating with same params."""
    from sklearn.base import clone
    return clone(model)


# =========================================================================
# Public API
# =========================================================================

def get_ml_projection(market: str, logs: pd.DataFrame,
                      proj_minutes: float,
                      home_flag: bool = False,
                      rest_days: int = 2,
                      playoff_flag: bool = False,
                      opp_abbr: str = "",
                      opp_def_rating: float = 1.0,
                      pace_factor: float = 1.0,
                      opp_pace: float = 1.0,
                      opp_rebound_pct: float = 1.0,
                      opp_pts_paint: float = 1.0,
                      travel_miles: float = 0.0,
                      tz_shift_hours: int = 0,
                      altitude_flag: bool = False,
                      # BDL season profile features
                      real_usage_pct: float = 0.0,
                      avg_touches: float = 0.0,
                      pnr_bh_freq: float = 0.0,
                      pnr_roll_freq: float = 0.0,
                      iso_freq: float = 0.0,
                      spotup_freq: float = 0.0,
                      transition_freq: float = 0.0,
                      postup_freq: float = 0.0,
                      drives_per_game: float = 0.0,
                      ts_pct: float = 0.0,
                      # BDL V2 advanced tracking features
                      avg_speed: float = 0.0,
                      avg_contested_fg_pct: float = 0.0,
                      avg_deflections: float = 0.0,
                      avg_points_paint: float = 0.0,
                      avg_pct_pts_paint: float = 0.0,
                      player_foul_rate: Optional[float] = None, feature_sink=None) -> Optional[float]:
    """
    Public API: return ML mean projection for a player/market.
    Returns None if model is untrained or data insufficient.

    Tries ensemble first; falls back to standalone XGBoost.
    """
    if proj_minutes <= 0:
        return None

    _bdl_kwargs = dict(
        real_usage_pct=real_usage_pct, avg_touches=avg_touches,
        pnr_bh_freq=pnr_bh_freq, pnr_roll_freq=pnr_roll_freq,
        iso_freq=iso_freq, spotup_freq=spotup_freq,
        transition_freq=transition_freq, postup_freq=postup_freq,
        drives_per_game=drives_per_game, ts_pct=ts_pct,
        avg_speed=avg_speed, avg_contested_fg_pct=avg_contested_fg_pct,
        avg_deflections=avg_deflections, avg_points_paint=avg_points_paint,
        avg_pct_pts_paint=avg_pct_pts_paint,
        player_foul_rate=player_foul_rate,
    )
    _travel_kwargs = dict(travel_miles=travel_miles, tz_shift_hours=tz_shift_hours,
                          altitude_flag=altitude_flag)

    if market == 'player_points_rebounds_assists':
        pts = get_ml_projection('player_points',  logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        reb = get_ml_projection('player_rebounds', logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        ast = get_ml_projection('player_assists',  logs, proj_minutes, home_flag, rest_days, playoff_flag, opp_abbr,
                                opp_def_rating, pace_factor, opp_pace, opp_rebound_pct, opp_pts_paint,
                                **_travel_kwargs, **_bdl_kwargs)
        if any(v is None for v in [pts, reb, ast]):
            return None
        return pts + reb + ast

    # Build features once, try ensemble then XGB fallback
    prop_model = PropMLModel(market)
    feats = prop_model.build_features(logs, home_flag, rest_days, playoff_flag, opp_abbr, opp_def_rating, pace_factor,
                                      opp_pace, opp_rebound_pct, opp_pts_paint,
                                      **_travel_kwargs, **_bdl_kwargs)
    if feats is None:
        return None
    if feature_sink is not None:
        feature_sink.update({name: float(feats[name]) for name in FEATURE_NAMES})

    # Try ensemble first
    ens = EnsembleModel(market)
    rate = ens.predict_rate(feats)

    # Fallback to standalone XGB
    if rate is None:
        rate = prop_model.predict_rate(feats)

    if rate is None:
        return None

    return max(0.0, rate * proj_minutes)


# =========================================================================
# Training functions
# =========================================================================

def train_models_from_logs(player_logs_list: List[pd.DataFrame],
                           opp_stats_df=None, team_stats_df=None, def_stats_df=None,
                           league_avg_pace=99.0) -> Dict[str, bool]:
    """Stage candidates using past-only features; current team tables are ignored."""
    results = {}
    for market in _MARKET_COL:
        prop, ensemble = PropMLModel(market), EnsembleModel(market)
        all_X, all_y, all_metadata = [], [], []
        for logs in player_logs_list:
            X, y, metadata = prop.build_training_data(logs, return_metadata=True)
            all_X.extend(X)
            all_y.extend(y)
            all_metadata.extend(metadata)
        ens_ok = ensemble.train(all_X, all_y, metadata=all_metadata)
        xgb_ok = prop.train(all_X, all_y, metadata=all_metadata)
        results[market] = ens_ok or xgb_ok
    return results


def _norm_player_name(name: str) -> str:
    """Lowercase + strip accents + normalise whitespace for CLV key matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    return ' '.join(nfkd.encode('ascii', 'ignore').decode('ascii').lower().split())


def build_clv_weight_lookup(db) -> Dict:
    """
    Query clv_tracking for all settled records and return a sample-weight dict.
    """
    try:
        with db.get_conn() as conn:
            rows = conn.execute(
                """
                SELECT ct.player_id                                       AS player,
                       ct.market,
                       ct.clv,
                       COALESCE(a.game_date, DATE(ct.alert_time))         AS game_date
                FROM   clv_tracking ct
                LEFT   JOIN alerts_sent a
                            ON  a.player_name = ct.player_id
                            AND a.market      = ct.market
                            AND date(a.timestamp) = date(ct.alert_time)
                WHERE  ct.clv IS NOT NULL
                """
            ).fetchall()
    except Exception:
        return {}

    lookup: Dict = {}
    for row in rows:
        player_n = _norm_player_name(str(row['player']))
        market   = str(row['market'])
        gdate    = str(row['game_date'] or '')
        clv_val  = float(row['clv'])

        if clv_val > 0.02:
            weight = 1.8
        elif clv_val > 0.0:
            weight = 1.2
        elif clv_val > -0.02:
            weight = 0.9
        else:
            weight = 0.6

        lookup[(player_n, market, gdate)] = weight

    return lookup


def train_models_with_clv_feedback(player_named_logs, db, opp_stats_df=None,
                                   team_stats_df=None, def_stats_df=None, league_avg_pace=99.0):
    """Compatibility entrypoint: outcome-derived CLV weights are disabled.

    This optional weighting has no validated incremental benefit. Evaluation
    must not be reweighted by closing outcomes from its own held-out period.
    """
    from src.utils.logging_utils import get_logger
    get_logger(__name__).warning('CLV sample weighting disabled; staging chronological candidates')
    return train_models_from_logs([logs for _, logs in player_named_logs])


def _build_abbr_lookup(df: pd.DataFrame, value_col: str) -> Dict[str, float]:
    """Build {team_abbr_upper: value} from a DataFrame."""
    result: Dict[str, float] = {}
    if 'TEAM_ABBREVIATION' in df.columns:
        for _, row in df.iterrows():
            abbr = str(row['TEAM_ABBREVIATION']).upper()
            result[abbr] = float(row[value_col])
    elif 'TEAM_NAME' in df.columns:
        try:
            from nba_api.stats.static import teams as _st
            name_to_abbr = {t['full_name'].lower(): t['abbreviation'].upper() for t in _st.get_teams()}
            for _, row in df.iterrows():
                abbr = name_to_abbr.get(str(row['TEAM_NAME']).lower(), '')
                if abbr:
                    result[abbr] = float(row[value_col])
        except Exception:
            pass
    return result
