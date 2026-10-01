import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MASTER_FILE = DATA_DIR / "goalie_stack_games_master.csv"
MODEL_ARTIFACT_PATH = DATA_DIR / "stack_predictive_model.json"
GAMECENTER_LANDING_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/landing"

ROLLING_WINDOWS = (5, 10)
EWM_ALPHA = 0.35

TEAM_PRIOR_FIELDS = ["gf", "ga", "sf", "sa", "goalie_fp", "game_total_goals", "game_total_shots", "game_stack_fp"]
GOALIE_PRIOR_FIELDS = [
    ("goalie_fp", "goalie_fp"),
    ("saves", "goalie_saves"),
    ("shots_against", "goalie_shots_against"),
    ("save_pct", "goalie_save_pct"),
]
# How many league-average games to blend into last season's numbers when seeding a new season.
# Shot volume carries over year to year (r ~0.6); goals and goalie results much less (r ~0.4 / ~0.15).
TEAM_CARRYOVER_SHRINK = {"sf": 50.0, "sa": 50.0, "game_total_shots": 50.0}
TEAM_CARRYOVER_SHRINK_DEFAULT = 120.0
GOALIE_CARRYOVER_SHRINK = 200.0
# Goalies with no earlier NHL games start from how new goalies have performed over their first starts.
ROOKIE_PRIOR_STARTS = 20
ROOKIE_PRIOR_MIN_SAMPLES = 50

# The model uses a small core of features. With ~4,000 noisy games, the full ~200-feature set
# plus team one-hots overfit and did no better than predicting the league average on 2025-26.
MODEL_FEATURES = [
    "season_expected_goals",
    "season_expected_shots",
    "ewm_expected_goals",
    "ewm_expected_shots",
    "venue_expected_goals",
    "venue_expected_shots",
    "season_pace_mean",
    "season_stack_env_mean",
    "goalie_fp_sum",
    "goalie_fp_form_sum",
    "goalie_save_pct_mean",
    "goalie_save_pct_form_mean",
    "away_home_goalie_fp_form_sum",
    "goalie_shots_against_mean",
    "rest_sum",
    "back_to_back_count",
    "goalie_start_share_sum",
]
USE_TEAM_ONE_HOTS = False
CANDIDATE_ALPHAS = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
# Tiers are percentiles of out-of-sample predictions: CORE = top 10% of games, STRONG = next 15%, etc.
TIER_PERCENTILES = [("CORE STACK", 90), ("STRONG STACK", 75), ("THIN EDGE STACK", 55), ("LEAN STACK", 40)]
TIER_WINDOW_GAMES = 500
CV_FOLD_STARTS = (0.5, 0.6, 0.7, 0.8, 0.9)
CV_FOLD_SIZE = 0.1
# Starting-goalie choice is a softmax over these scores. Weights were fit as a conditional logit on
# 2024-25 and 2025-26 regular-season starts (fit on 2024-25 alone, they cut 2025-26 log loss from
# 0.82 to 0.77 vs the old hand-set weights, and from 0.91 to 0.72 over each team's first 10 games).
STARTER_WEIGHTS = {
    "season_share": 1.327,
    "form5_share": 2.709,
    "started_last": -0.567,
    "back_to_back_started_last": -2.4,
}
STARTER_PRIOR_GAMES = 5.0


class PredictiveStackModelError(RuntimeError):
    pass


def load_master_games(master_file=MASTER_FILE):
    path = Path(master_file)
    if not path.exists():
        raise PredictiveStackModelError(f"Master file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "game_id",
        "date",
        "season",
        "away_team",
        "home_team",
        "away_goalie",
        "home_goalie",
        "away_goalie_id",
        "home_goalie_id",
        "away_score",
        "home_score",
        "away_saves",
        "home_saves",
        "away_shots_against",
        "home_shots_against",
        "away_fp",
        "home_fp",
        "total_goals",
        "total_shots",
        "stack_fp",
    }
    missing = required - set(df.columns)
    if missing:
        raise PredictiveStackModelError(f"Master file missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], format="mixed").dt.normalize()
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def _iter_team_records(games_df):
    for game in games_df.itertuples(index=False):
        game_date = pd.Timestamp(game.date)
        yield {
            "season": game.season,
            "team": game.away_team,
            "date": game_date,
            "is_home": 0,
            "gf": float(game.away_score),
            "ga": float(game.home_score),
            "sf": float(game.home_shots_against),
            "sa": float(game.away_shots_against),
            "goalie_fp": float(game.away_fp),
            "game_total_goals": float(game.total_goals),
            "game_total_shots": float(game.total_shots),
            "game_stack_fp": float(game.stack_fp),
        }
        yield {
            "season": game.season,
            "team": game.home_team,
            "date": game_date,
            "is_home": 1,
            "gf": float(game.home_score),
            "ga": float(game.away_score),
            "sf": float(game.away_shots_against),
            "sa": float(game.home_shots_against),
            "goalie_fp": float(game.home_fp),
            "game_total_goals": float(game.total_goals),
            "game_total_shots": float(game.total_shots),
            "game_stack_fp": float(game.stack_fp),
        }


def _iter_goalie_records(games_df):
    for game in games_df.itertuples(index=False):
        game_date = pd.Timestamp(game.date)
        away_shots_against = float(game.away_shots_against)
        home_shots_against = float(game.home_shots_against)

        yield {
            "season": game.season,
            "team": game.away_team,
            "goalie_id": int(game.away_goalie_id),
            "goalie_name": game.away_goalie,
            "date": game_date,
            "is_home": 0,
            "goalie_fp": float(game.away_fp),
            "saves": float(game.away_saves),
            "shots_against": away_shots_against,
            "save_pct": float(game.away_saves) / away_shots_against if away_shots_against > 0 else 0.0,
        }
        yield {
            "season": game.season,
            "team": game.home_team,
            "goalie_id": int(game.home_goalie_id),
            "goalie_name": game.home_goalie,
            "date": game_date,
            "is_home": 1,
            "goalie_fp": float(game.home_fp),
            "saves": float(game.home_saves),
            "shots_against": home_shots_against,
            "save_pct": float(game.home_saves) / home_shots_against if home_shots_against > 0 else 0.0,
        }


def compute_priors(games_df):
    records = list(_iter_team_records(games_df))
    goalie_records = list(_iter_goalie_records(games_df))
    priors = {}
    for field in TEAM_PRIOR_FIELDS:
        priors[field] = float(pd.Series(record[field] for record in records).mean())
    for field in ["saves", "shots_against", "save_pct"]:
        priors[f"goalie_{field}"] = float(pd.Series(record[field] for record in goalie_records).mean())
    priors["rest_days"] = 3.0
    priors["games_played"] = 10.0

    # Goalies first seen after the dataset's opening season are treated as new to the league.
    goalie_df = pd.DataFrame(goalie_records)
    first_season = goalie_df.groupby("goalie_id")["season"].transform("min")
    start_number = goalie_df.groupby("goalie_id").cumcount()
    rookies = goalie_df[(first_season > goalie_df["season"].min()) & (start_number < ROOKIE_PRIOR_STARTS)]
    for field, prior_key in GOALIE_PRIOR_FIELDS:
        if len(rookies) >= ROOKIE_PRIOR_MIN_SAMPLES:
            priors[f"rookie_{prior_key}"] = float(rookies[field].mean())
        else:
            priors[f"rookie_{prior_key}"] = priors[prior_key]
    return priors


def season_for_date(date):
    """NHL season code (e.g. 20262027) for a date; seasons roll over in August."""
    ts = pd.Timestamp(date)
    start_year = ts.year if ts.month >= 8 else ts.year - 1
    return start_year * 10000 + start_year + 1


def previous_season(season):
    return int(season) - 10001


def _shrink_toward(values, prior, shrink):
    return float((values.sum() + shrink * prior) / (len(values) + shrink))


def compute_carryover_priors(games_df, priors, extra_seasons=()):
    """Per-season starting points for teams and goalies, built only from earlier seasons."""
    team_records = pd.DataFrame(_iter_team_records(games_df))
    goalie_records = pd.DataFrame(_iter_goalie_records(games_df))
    seasons = sorted(set(games_df["season"].astype(int)) | {int(season) for season in extra_seasons})
    team_priors = {}
    goalie_priors = {}
    seasons_with_history = set()

    for season in seasons:
        last_season = team_records[team_records["season"] == previous_season(season)]
        for team, group in last_season.groupby("team"):
            team_priors[(season, team)] = {
                field: _shrink_toward(
                    group[field],
                    priors[field],
                    TEAM_CARRYOVER_SHRINK.get(field, TEAM_CARRYOVER_SHRINK_DEFAULT),
                )
                for field in TEAM_PRIOR_FIELDS
            }

        earlier = goalie_records[goalie_records["season"] < season]
        if earlier.empty:
            continue
        seasons_with_history.add(season)
        for goalie_id, group in earlier.groupby("goalie_id"):
            goalie_priors[(season, int(goalie_id))] = {
                prior_key: _shrink_toward(group[field], priors[prior_key], GOALIE_CARRYOVER_SHRINK)
                for field, prior_key in GOALIE_PRIOR_FIELDS
            }

    return {"team": team_priors, "goalie": goalie_priors, "seasons_with_history": seasons_with_history}


def team_priors_for(carryover, priors, season, team):
    overrides = carryover["team"].get((int(season), team)) if carryover else None
    return {**priors, **overrides} if overrides else priors


def goalie_priors_for(carryover, priors, season, goalie_id):
    if not carryover or goalie_id is None:
        return priors
    overrides = carryover["goalie"].get((int(season), int(goalie_id)))
    if overrides:
        return {**priors, **overrides}
    if int(season) in carryover["seasons_with_history"]:
        return {**priors, **{key: priors[f"rookie_{key}"] for _, key in GOALIE_PRIOR_FIELDS}}
    return priors


def _smoothed_mean(values, prior, shrink=6.0):
    if not values:
        return float(prior)
    return float((sum(values) + shrink * prior) / (len(values) + shrink))


def _exp_weighted_mean(values, prior, alpha=EWM_ALPHA):
    if not values:
        return float(prior)
    result = float(prior)
    for value in values:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def summarize_team_history(history, game_date, current_is_home, priors):
    gp = len(history)
    last_date = history[-1]["date"] if history else None
    rest_days = priors["rest_days"] if last_date is None else max(1.0, float((game_date - last_date).days))
    back_to_back = 1.0 if rest_days <= 1.0 else 0.0

    def values(field, items):
        return [float(item[field]) for item in items]

    recent5 = history[-5:]
    recent10 = history[-10:]
    venue_history = [item for item in history if item["is_home"] == current_is_home]

    summary = {
        "gp_prior": float(gp),
        "rest_days": rest_days,
        "back_to_back": back_to_back,
        "venue_switch": 0.0 if not history else float(history[-1]["is_home"] != current_is_home),
        "same_venue_streak": 0.0,
    }

    if history:
        streak = 0
        for item in reversed(history):
            if item["is_home"] == history[-1]["is_home"]:
                streak += 1
            else:
                break
        summary["same_venue_streak"] = float(streak)

    for field in TEAM_PRIOR_FIELDS:
        summary[f"season_{field}"] = _smoothed_mean(values(field, history), priors[field], shrink=8.0)
        summary[f"form5_{field}"] = _smoothed_mean(values(field, recent5), priors[field], shrink=3.0)
        summary[f"form10_{field}"] = _smoothed_mean(values(field, recent10), priors[field], shrink=4.0)
        summary[f"ewm_{field}"] = _exp_weighted_mean(values(field, history), priors[field], alpha=EWM_ALPHA)
        summary[f"venue_{field}"] = _smoothed_mean(values(field, venue_history), priors[field], shrink=5.0)

    return summary


def summarize_goalie_history(goalie_history, team_history, game_date, current_is_home, priors):
    gp = len(goalie_history)
    last_date = goalie_history[-1]["date"] if goalie_history else None
    rest_days = priors["rest_days"] if last_date is None else max(1.0, float((game_date - last_date).days))
    started_last_game = 0.0
    consecutive_team_starts = 0.0

    if team_history:
        started_last_game = float(team_history[-1].get("goalie_id") == (goalie_history[-1]["goalie_id"] if goalie_history else None))
        for item in reversed(team_history):
            if goalie_history and item.get("goalie_id") == goalie_history[-1]["goalie_id"]:
                consecutive_team_starts += 1.0
            else:
                break

    def values(field, items):
        return [float(item[field]) for item in items]

    recent5 = goalie_history[-5:]
    recent10 = goalie_history[-10:]
    venue_history = [item for item in goalie_history if item["is_home"] == current_is_home]
    team_games = max(len(team_history), 1)
    goalie_id = goalie_history[-1]["goalie_id"] if goalie_history else None

    def recent_start_share(n):
        # Share of the team's last n games this goalie started (not his own last n starts).
        recent_team = team_history[-n:]
        if not recent_team or goalie_id is None:
            return 0.0
        return sum(item.get("goalie_id") == goalie_id for item in recent_team) / len(recent_team)

    summary = {
        "gp": float(gp),
        "rest_days": rest_days,
        "started_last_game": started_last_game,
        "consecutive_starts": consecutive_team_starts,
        "season_start_share": float(gp / team_games),
        "form5_start_share": float(recent_start_share(5)),
        "form10_start_share": float(recent_start_share(10)),
    }

    for field, prior_key in GOALIE_PRIOR_FIELDS:
        prior = priors[prior_key]
        summary[f"season_{field}"] = _smoothed_mean(values(field, goalie_history), prior, shrink=6.0)
        summary[f"form5_{field}"] = _smoothed_mean(values(field, recent5), prior, shrink=3.0)
        summary[f"form10_{field}"] = _smoothed_mean(values(field, recent10), prior, shrink=4.0)
        summary[f"ewm_{field}"] = _exp_weighted_mean(values(field, goalie_history), prior, alpha=EWM_ALPHA)
        summary[f"venue_{field}"] = _smoothed_mean(values(field, venue_history), prior, shrink=4.0)

    return summary


def build_matchup_feature_row(
    away_team,
    home_team,
    game_date,
    away_history,
    home_history,
    priors,
    away_goalie_summary=None,
    home_goalie_summary=None,
    away_team_priors=None,
    home_team_priors=None,
):
    away_team_priors = away_team_priors or priors
    home_team_priors = home_team_priors or priors
    away = summarize_team_history(away_history, game_date, current_is_home=0, priors=away_team_priors)
    home = summarize_team_history(home_history, game_date, current_is_home=1, priors=home_team_priors)

    row = {
        "away_team": away_team,
        "home_team": home_team,
        "date": pd.Timestamp(game_date),
    }

    for prefix, summary in [("away", away), ("home", home)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = float(value)

    row["gp_min"] = min(away["gp_prior"], home["gp_prior"])
    row["gp_max"] = max(away["gp_prior"], home["gp_prior"])
    row["rest_sum"] = away["rest_days"] + home["rest_days"]
    row["rest_diff"] = away["rest_days"] - home["rest_days"]
    row["back_to_back_count"] = away["back_to_back"] + home["back_to_back"]
    row["venue_switch_sum"] = away["venue_switch"] + home["venue_switch"]
    row["same_venue_streak_mean"] = 0.5 * (away["same_venue_streak"] + home["same_venue_streak"])

    for prefix in ["season", "form5", "form10", "ewm", "venue"]:
        away_gf = away[f"{prefix}_gf"]
        away_ga = away[f"{prefix}_ga"]
        away_sf = away[f"{prefix}_sf"]
        away_sa = away[f"{prefix}_sa"]
        home_gf = home[f"{prefix}_gf"]
        home_ga = home[f"{prefix}_ga"]
        home_sf = home[f"{prefix}_sf"]
        home_sa = home[f"{prefix}_sa"]

        exp_goals = 0.5 * (away_gf + home_ga + home_gf + away_ga)
        exp_shots = 0.5 * (away_sf + home_sa + home_sf + away_sa)
        row[f"{prefix}_expected_goals"] = exp_goals
        row[f"{prefix}_expected_shots"] = exp_shots
        row[f"{prefix}_goal_shot_interaction"] = exp_goals * exp_shots
        row[f"{prefix}_expected_goals_sq"] = exp_goals ** 2
        row[f"{prefix}_expected_shots_sq"] = exp_shots ** 2
        row[f"{prefix}_pace_mean"] = 0.5 * (
            away[f"{prefix}_game_total_shots"] + home[f"{prefix}_game_total_shots"]
        )
        row[f"{prefix}_stack_env_mean"] = 0.5 * (
            away[f"{prefix}_game_stack_fp"] + home[f"{prefix}_game_stack_fp"]
        )

    row["away_home_goalie_fp_sum"] = away["season_goalie_fp"] + home["season_goalie_fp"]
    row["away_home_goalie_fp_form_sum"] = away["form10_goalie_fp"] + home["form10_goalie_fp"]
    row["away_home_goalie_fp_ewm_sum"] = away["ewm_goalie_fp"] + home["ewm_goalie_fp"]

    away_goalie_summary = away_goalie_summary or summarize_goalie_history([], away_history, game_date, 0, away_team_priors)
    home_goalie_summary = home_goalie_summary or summarize_goalie_history([], home_history, game_date, 1, home_team_priors)

    for prefix, summary in [("away_goalie", away_goalie_summary), ("home_goalie", home_goalie_summary)]:
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = float(value)

    row["goalie_start_share_sum"] = (
        away_goalie_summary["season_start_share"] + home_goalie_summary["season_start_share"]
    )
    row["goalie_start_share_form_sum"] = (
        away_goalie_summary["form10_start_share"] + home_goalie_summary["form10_start_share"]
    )
    row["goalie_fp_sum"] = away_goalie_summary["season_goalie_fp"] + home_goalie_summary["season_goalie_fp"]
    row["goalie_fp_form_sum"] = away_goalie_summary["form10_goalie_fp"] + home_goalie_summary["form10_goalie_fp"]
    row["goalie_save_pct_mean"] = 0.5 * (
        away_goalie_summary["season_save_pct"] + home_goalie_summary["season_save_pct"]
    )
    row["goalie_save_pct_form_mean"] = 0.5 * (
        away_goalie_summary["form10_save_pct"] + home_goalie_summary["form10_save_pct"]
    )
    row["goalie_rest_diff"] = away_goalie_summary["rest_days"] - home_goalie_summary["rest_days"]
    row["goalie_rest_sum"] = away_goalie_summary["rest_days"] + home_goalie_summary["rest_days"]
    row["goalie_started_last_game_count"] = (
        away_goalie_summary["started_last_game"] + home_goalie_summary["started_last_game"]
    )
    row["goalie_consecutive_starts_sum"] = (
        away_goalie_summary["consecutive_starts"] + home_goalie_summary["consecutive_starts"]
    )
    row["goalie_shots_against_mean"] = 0.5 * (
        away_goalie_summary["season_shots_against"] + home_goalie_summary["season_shots_against"]
    )

    return row


def build_training_dataframe(games_df):
    priors = compute_priors(games_df)
    carryover = compute_carryover_priors(games_df, priors)
    histories = defaultdict(list)
    goalie_histories = defaultdict(list)
    rows = []

    for game in games_df.itertuples(index=False):
        away_key = (game.season, game.away_team)
        home_key = (game.season, game.home_team)
        game_date = pd.Timestamp(game.date)
        away_goalie_key = (game.season, game.away_team, int(game.away_goalie_id))
        home_goalie_key = (game.season, game.home_team, int(game.home_goalie_id))
        away_goalie_summary = summarize_goalie_history(
            goalie_histories[away_goalie_key],
            histories[away_key],
            game_date,
            current_is_home=0,
            priors=goalie_priors_for(carryover, priors, game.season, game.away_goalie_id),
        )
        home_goalie_summary = summarize_goalie_history(
            goalie_histories[home_goalie_key],
            histories[home_key],
            game_date,
            current_is_home=1,
            priors=goalie_priors_for(carryover, priors, game.season, game.home_goalie_id),
        )

        feature_row = build_matchup_feature_row(
            away_team=game.away_team,
            home_team=game.home_team,
            game_date=game_date,
            away_history=histories[away_key],
            home_history=histories[home_key],
            priors=priors,
            away_goalie_summary=away_goalie_summary,
            home_goalie_summary=home_goalie_summary,
            away_team_priors=team_priors_for(carryover, priors, game.season, game.away_team),
            home_team_priors=team_priors_for(carryover, priors, game.season, game.home_team),
        )
        feature_row["season"] = int(game.season)
        feature_row["game_id"] = int(game.game_id)
        feature_row["stack_fp"] = float(game.stack_fp)
        feature_row["target_total_goals"] = float(game.total_goals)
        feature_row["target_total_shots"] = float(game.total_shots)
        rows.append(feature_row)

        histories[away_key].append(
            {
                "date": game_date,
                "is_home": 0,
                "gf": float(game.away_score),
                "ga": float(game.home_score),
                "sf": float(game.home_shots_against),
                "sa": float(game.away_shots_against),
                "goalie_fp": float(game.away_fp),
                "game_total_goals": float(game.total_goals),
                "game_total_shots": float(game.total_shots),
                "game_stack_fp": float(game.stack_fp),
                "goalie_id": int(game.away_goalie_id),
            }
        )
        histories[home_key].append(
            {
                "date": game_date,
                "is_home": 1,
                "gf": float(game.home_score),
                "ga": float(game.away_score),
                "sf": float(game.away_shots_against),
                "sa": float(game.home_shots_against),
                "goalie_fp": float(game.home_fp),
                "game_total_goals": float(game.total_goals),
                "game_total_shots": float(game.total_shots),
                "game_stack_fp": float(game.stack_fp),
                "goalie_id": int(game.home_goalie_id),
            }
        )
        goalie_histories[away_goalie_key].append(
            {
                "date": game_date,
                "is_home": 0,
                "goalie_id": int(game.away_goalie_id),
                "goalie_fp": float(game.away_fp),
                "saves": float(game.away_saves),
                "shots_against": float(game.away_shots_against),
                "save_pct": float(game.away_saves) / float(game.away_shots_against)
                if float(game.away_shots_against) > 0
                else 0.0,
            }
        )
        goalie_histories[home_goalie_key].append(
            {
                "date": game_date,
                "is_home": 1,
                "goalie_id": int(game.home_goalie_id),
                "goalie_fp": float(game.home_fp),
                "saves": float(game.home_saves),
                "shots_against": float(game.home_shots_against),
                "save_pct": float(game.home_saves) / float(game.home_shots_against)
                if float(game.home_shots_against) > 0
                else 0.0,
            }
        )

    training_df = pd.DataFrame(rows).sort_values(["date", "game_id"]).reset_index(drop=True)
    return training_df, priors


def feature_columns(training_df):
    missing = [column for column in MODEL_FEATURES if column not in training_df.columns]
    if missing:
        raise PredictiveStackModelError(f"Training data missing model features: {missing}")
    return list(MODEL_FEATURES)


def build_design_matrix(training_df, numeric_columns, team_codes):
    base = training_df[numeric_columns].to_numpy(dtype=float)
    if not team_codes:
        return base, list(numeric_columns)
    away_matrix = np.column_stack(
        [(training_df["away_team"] == code).to_numpy(dtype=float) for code in team_codes]
    )
    home_matrix = np.column_stack(
        [(training_df["home_team"] == code).to_numpy(dtype=float) for code in team_codes]
    )
    design_columns = (
        list(numeric_columns)
        + [f"away_team__{code}" for code in team_codes]
        + [f"home_team__{code}" for code in team_codes]
    )
    return np.column_stack([base, away_matrix, home_matrix]), design_columns


def _fit_ridge(X, y, alpha):
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = float(y.mean())

    Xs = (X - x_mean) / x_std
    yc = y - y_mean
    penalty = alpha * np.eye(Xs.shape[1])
    beta = np.linalg.solve(Xs.T @ Xs + penalty, Xs.T @ yc)

    return {
        "alpha": float(alpha),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "beta": beta,
    }


def _predict_ridge(model, X):
    Xs = (X - model["x_mean"]) / model["x_std"]
    return model["y_mean"] + Xs @ model["beta"]


def _fit_stack_surface(goals, shots, stack_fp):
    X = np.column_stack(
        [
            np.ones(len(goals)),
            goals,
            goals ** 2,
            shots,
            goals * shots,
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(X, stack_fp, rcond=None)
    return {
        "intercept": float(coefficients[0]),
        "b_goals": float(coefficients[1]),
        "b_goals_sq": float(coefficients[2]),
        "b_shots": float(coefficients[3]),
        "b_interaction": float(coefficients[4]),
    }


def _predict_stack_surface(surface, goals, shots):
    return (
        surface["intercept"]
        + surface["b_goals"] * goals
        + surface["b_goals_sq"] * (goals ** 2)
        + surface["b_shots"] * shots
        + surface["b_interaction"] * (goals * shots)
    )


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _spearman(y_true, y_pred):
    true_rank = pd.Series(y_true).rank(method="average")
    pred_rank = pd.Series(y_pred).rank(method="average")
    return float(true_rank.corr(pred_rank))


def fit_predictive_stack_model(training_df):
    if len(training_df) < 500:
        raise PredictiveStackModelError("Need at least 500 historical games to fit the predictive model.")

    numeric_columns = feature_columns(training_df)
    team_codes = (
        sorted(set(training_df["away_team"]).union(set(training_df["home_team"]))) if USE_TEAM_ONE_HOTS else []
    )
    X, design_columns = build_design_matrix(training_df, numeric_columns, team_codes)
    y_stack = training_df["stack_fp"].to_numpy(dtype=float)
    y_goals = training_df["target_total_goals"].to_numpy(dtype=float)
    y_shots = training_df["target_total_shots"].to_numpy(dtype=float)

    n = len(training_df)
    # Rolling-origin folds: train on everything before each cut, score the next 10% of games.
    folds = [(int(n * start), int(n * (start + CV_FOLD_SIZE))) for start in CV_FOLD_STARTS]
    if folds[0][0] < 100:
        raise PredictiveStackModelError("Not enough historical rows for chronological validation folds.")

    def choose_alpha(y):
        fold_maes = {
            alpha: np.mean(
                [_mae(y[end:stop], _predict_ridge(_fit_ridge(X[:end], y[:end], alpha=alpha), X[end:stop])) for end, stop in folds]
            )
            for alpha in CANDIDATE_ALPHAS
        }
        return min(fold_maes, key=fold_maes.get)

    direct_alpha = choose_alpha(y_stack)
    goals_alpha = choose_alpha(y_goals)
    shots_alpha = choose_alpha(y_shots)

    # Out-of-sample direct and structural (goals + shots -> stack FP) predictions for every fold.
    fold_direct, fold_structural, fold_actual = [], [], []
    for end, stop in folds:
        fold_direct.append(_predict_ridge(_fit_ridge(X[:end], y_stack[:end], alpha=direct_alpha), X[end:stop]))
        goals_preds = np.clip(_predict_ridge(_fit_ridge(X[:end], y_goals[:end], alpha=goals_alpha), X[end:stop]), 0.0, None)
        shots_preds = np.clip(_predict_ridge(_fit_ridge(X[:end], y_shots[:end], alpha=shots_alpha), X[end:stop]), 0.0, None)
        surface = _fit_stack_surface(y_goals[:end], y_shots[:end], y_stack[:end])
        fold_structural.append(_predict_stack_surface(surface, goals_preds, shots_preds))
        fold_actual.append(y_stack[end:stop])
    direct_oos = np.concatenate(fold_direct)
    structural_oos = np.concatenate(fold_structural)
    actual_oos = np.concatenate(fold_actual)

    best_weight = None
    best_weight_mae = None
    for weight in np.linspace(0.0, 1.0, 11):
        mae = _mae(actual_oos, weight * direct_oos + (1.0 - weight) * structural_oos)
        if best_weight_mae is None or mae < best_weight_mae:
            best_weight_mae = mae
            best_weight = float(weight)

    oos_preds = best_weight * direct_oos + (1.0 - best_weight) * structural_oos
    last_fold_size = len(fold_actual[-1])
    validation_metrics = {
        "mae": _mae(actual_oos, oos_preds),
        "rmse": _rmse(actual_oos, oos_preds),
        "spearman": _spearman(actual_oos, oos_preds),
        "baseline_mae": float(
            np.mean(np.concatenate([np.abs(y_stack[end:stop] - y_stack[:end].mean()) for end, stop in folds]))
        ),
    }
    test_metrics = {
        "mae": _mae(actual_oos[-last_fold_size:], oos_preds[-last_fold_size:]),
        "rmse": _rmse(actual_oos[-last_fold_size:], oos_preds[-last_fold_size:]),
        "spearman": _spearman(actual_oos[-last_fold_size:], oos_preds[-last_fold_size:]),
    }
    residual_std = float(np.std(actual_oos - oos_preds, ddof=1))

    direct_final_model = _fit_ridge(X, y_stack, alpha=direct_alpha)
    goals_final_model = _fit_ridge(X, y_goals, alpha=goals_alpha)
    shots_final_model = _fit_ridge(X, y_shots, alpha=shots_alpha)
    final_surface = _fit_stack_surface(y_goals, y_shots, y_stack)

    # Cut tiers from the final model's predictions on the most recent games, so they track
    # the current scoring environment instead of older seasons.
    recent_X = X[-TIER_WINDOW_GAMES:]
    recent_goals = np.clip(_predict_ridge(goals_final_model, recent_X), 0.0, None)
    recent_shots = np.clip(_predict_ridge(shots_final_model, recent_X), 0.0, None)
    recent_preds = best_weight * _predict_ridge(direct_final_model, recent_X) + (1.0 - best_weight) * _predict_stack_surface(
        final_surface, recent_goals, recent_shots
    )
    tier_thresholds = [
        {"tier": tier, "min_predicted_stack_fp": float(np.percentile(recent_preds, pct))}
        for tier, pct in TIER_PERCENTILES
    ]
    train_end = folds[0][0]
    val_end = n

    artifact = {
        "model_type": "ensemble_stack_model",
        "ensemble_weight_direct": best_weight,
        "feature_columns": design_columns,
        "numeric_feature_columns": numeric_columns,
        "team_codes": team_codes,
        "direct_alpha": float(direct_alpha),
        "goals_alpha": float(goals_alpha),
        "shots_alpha": float(shots_alpha),
        "x_mean": direct_final_model["x_mean"].tolist(),
        "x_std": direct_final_model["x_std"].tolist(),
        "direct_y_mean": float(direct_final_model["y_mean"]),
        "direct_beta": direct_final_model["beta"].tolist(),
        "goals_y_mean": float(goals_final_model["y_mean"]),
        "goals_beta": goals_final_model["beta"].tolist(),
        "shots_y_mean": float(shots_final_model["y_mean"]),
        "shots_beta": shots_final_model["beta"].tolist(),
        "stack_surface": final_surface,
        "residual_std": residual_std,
        "training_rows": int(n),
        "train_end_date": str(training_df.iloc[train_end - 1]["date"].date()),
        "validation_end_date": str(training_df.iloc[val_end - 1]["date"].date()),
        "latest_training_date": str(training_df.iloc[-1]["date"].date()),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "tier_thresholds": tier_thresholds,
    }
    return artifact


def save_model_artifact(artifact, output_path=MODEL_ARTIFACT_PATH):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)


def load_model_artifact(artifact_path=MODEL_ARTIFACT_PATH):
    path = Path(artifact_path)
    if not path.exists():
        raise PredictiveStackModelError(f"Predictive model artifact not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_predictive_stack_model(master_file=MASTER_FILE, output_path=MODEL_ARTIFACT_PATH):
    games_df = load_master_games(master_file)
    training_df, priors = build_training_dataframe(games_df)
    artifact = fit_predictive_stack_model(training_df)
    artifact["priors"] = priors
    save_model_artifact(artifact, output_path=output_path)
    return artifact


def build_histories_through_date(games_df, target_date):
    histories = defaultdict(list)
    goalie_histories = defaultdict(list)
    cutoff = pd.Timestamp(target_date).normalize()

    for game in games_df.itertuples(index=False):
        if pd.Timestamp(game.date) >= cutoff:
            break

        away_key = (game.season, game.away_team)
        home_key = (game.season, game.home_team)
        game_date = pd.Timestamp(game.date)

        histories[away_key].append(
            {
                "date": game_date,
                "is_home": 0,
                "gf": float(game.away_score),
                "ga": float(game.home_score),
                "sf": float(game.home_shots_against),
                "sa": float(game.away_shots_against),
                "goalie_fp": float(game.away_fp),
                "game_total_goals": float(game.total_goals),
                "game_total_shots": float(game.total_shots),
                "game_stack_fp": float(game.stack_fp),
                "goalie_id": int(game.away_goalie_id),
            }
        )
        histories[home_key].append(
            {
                "date": game_date,
                "is_home": 1,
                "gf": float(game.home_score),
                "ga": float(game.away_score),
                "sf": float(game.away_shots_against),
                "sa": float(game.home_shots_against),
                "goalie_fp": float(game.home_fp),
                "game_total_goals": float(game.total_goals),
                "game_total_shots": float(game.total_shots),
                "game_stack_fp": float(game.stack_fp),
                "goalie_id": int(game.home_goalie_id),
            }
        )
        goalie_histories[(game.season, game.away_team, int(game.away_goalie_id))].append(
            {
                "date": game_date,
                "is_home": 0,
                "goalie_id": int(game.away_goalie_id),
                "goalie_fp": float(game.away_fp),
                "saves": float(game.away_saves),
                "shots_against": float(game.away_shots_against),
                "save_pct": float(game.away_saves) / float(game.away_shots_against)
                if float(game.away_shots_against) > 0
                else 0.0,
            }
        )
        goalie_histories[(game.season, game.home_team, int(game.home_goalie_id))].append(
            {
                "date": game_date,
                "is_home": 1,
                "goalie_id": int(game.home_goalie_id),
                "goalie_fp": float(game.home_fp),
                "saves": float(game.home_saves),
                "shots_against": float(game.home_shots_against),
                "save_pct": float(game.home_saves) / float(game.home_shots_against)
                if float(game.home_shots_against) > 0
                else 0.0,
            }
        )

    return histories, goalie_histories


def infer_target_season(games_df, target_date):
    # Based on the calendar, not the last game played, so opening day starts a fresh season.
    return season_for_date(target_date)


def _standard_normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fetch_gamecenter_landing(game_id, timeout=20):
    response = requests.get(GAMECENTER_LANDING_URL.format(game_id=game_id), timeout=timeout)
    response.raise_for_status()
    return response.json()


def extract_goalie_candidates(gamecenter_data, team_abbrev, team_id):
    matchup = gamecenter_data.get("matchup", {})
    goalie_stats = matchup.get("goalieSeasonStats", {}).get("goalies", [])
    team_candidates = [goalie for goalie in goalie_stats if goalie.get("teamId") == team_id]
    candidates = []
    for goalie in team_candidates:
        candidates.append(
            {
                "goalie_id": int(goalie["playerId"]),
                "goalie_name": goalie["name"]["default"],
                "games_played": float(goalie.get("gamesPlayed", 0)),
                "shots_against": float(goalie.get("shotsAgainst", 0)),
                "saves": float(goalie.get("saves", 0)),
                "save_pct": float(goalie.get("savePctg", 0)),
                "team": team_abbrev,
            }
        )
    return candidates


def infer_goalie_start_probabilities(
    season,
    team_code,
    team_history,
    goalie_histories,
    candidates,
    game_date,
    current_is_home,
    priors,
    carryover=None,
):
    if not candidates:
        return []

    team_back_to_back = 0.0
    if team_history:
        team_back_to_back = 1.0 if max(1.0, float((game_date - team_history[-1]["date"]).days)) <= 1.0 else 0.0
    team_games = float(len(team_history))
    recent_games = float(min(len(team_history), 5))
    total_preview_gp = max(sum(item["games_played"] for item in candidates), 1.0)

    scored_candidates = []
    for candidate in candidates:
        goalie_history = goalie_histories.get((season, team_code, candidate["goalie_id"]), [])
        summary = summarize_goalie_history(
            goalie_history,
            team_history,
            game_date,
            current_is_home,
            goalie_priors_for(carryover, priors, season, candidate["goalie_id"]),
        )

        # The preview's games played (last season's early on) seeds the start shares as pseudo-games,
        # so opening-week picks lean on last season's usage until this season's starts take over.
        preview_share = candidate["games_played"] / total_preview_gp
        season_share = (summary["gp"] + STARTER_PRIOR_GAMES * preview_share) / (team_games + STARTER_PRIOR_GAMES)
        form_share = (summary["form5_start_share"] * recent_games + 2.0 * preview_share) / (recent_games + 2.0)
        started_last = summary["started_last_game"]

        score = (
            STARTER_WEIGHTS["season_share"] * season_share
            + STARTER_WEIGHTS["form5_share"] * form_share
            + STARTER_WEIGHTS["started_last"] * started_last
            + STARTER_WEIGHTS["back_to_back_started_last"] * team_back_to_back * started_last
        )

        scored_candidates.append(
            {
                "goalie_id": candidate["goalie_id"],
                "goalie_name": candidate["goalie_name"],
                "summary": summary,
                "score": score,
            }
        )

    max_score = max(item["score"] for item in scored_candidates)
    weights = [math.exp(item["score"] - max_score) for item in scored_candidates]
    total_weight = sum(weights)

    results = []
    for item, weight in zip(scored_candidates, weights):
        results.append(
            {
                "goalie_id": item["goalie_id"],
                "goalie_name": item["goalie_name"],
                "summary": item["summary"],
                "start_probability": weight / total_weight if total_weight > 0 else 1.0 / len(scored_candidates),
            }
        )

    return sorted(results, key=lambda row: row["start_probability"], reverse=True)


def classify_stack_tier(predicted_stack_fp, prob_positive, prob_5_plus, tier_thresholds=None):
    if tier_thresholds:
        for threshold in tier_thresholds:
            if predicted_stack_fp >= threshold["min_predicted_stack_fp"]:
                return threshold["tier"]
        return "FADE"

    # Fixed cutoffs for artifacts trained before percentile tiers existed.
    if predicted_stack_fp >= 4.25 or prob_5_plus >= 0.42:
        return "CORE STACK"
    if predicted_stack_fp >= 3.0 or prob_5_plus >= 0.30:
        return "STRONG STACK"
    if predicted_stack_fp >= 2.0 or prob_5_plus >= 0.20:
        return "THIN EDGE STACK"
    if predicted_stack_fp >= 1.0 or prob_positive >= 0.60:
        return "LEAN STACK"
    return "FADE"


def evaluate_matchup_with_model(
    away_team,
    home_team,
    target_date,
    games_df,
    artifact,
    away_goalie_candidates=None,
    home_goalie_candidates=None,
    carryover=None,
):
    priors = artifact["priors"]
    season = infer_target_season(games_df, target_date)
    if carryover is None:
        carryover = compute_carryover_priors(games_df, priors, extra_seasons=[season])
    away_team_priors = team_priors_for(carryover, priors, season, away_team)
    home_team_priors = team_priors_for(carryover, priors, season, home_team)
    histories, goalie_histories = build_histories_through_date(games_df, target_date)
    game_date = pd.Timestamp(target_date).normalize()

    away_team_history = histories[(season, away_team)]
    home_team_history = histories[(season, home_team)]

    def build_goalie_summary(candidate, team, is_home, team_history):
        goalie_history = goalie_histories[(season, team, candidate["goalie_id"])]
        goalie_priors = goalie_priors_for(carryover, priors, season, candidate["goalie_id"])
        return summarize_goalie_history(goalie_history, team_history, game_date, is_home, goalie_priors)

    if away_goalie_candidates:
        away_goalies = [
            {
                "goalie_id": candidate["goalie_id"],
                "goalie_name": candidate["goalie_name"],
                "summary": build_goalie_summary(candidate, away_team, 0, away_team_history),
                "start_probability": float(candidate["start_probability"]),
            }
            for candidate in away_goalie_candidates
        ]
    else:
        away_goalies = [
            {
                "goalie_id": None,
                "goalie_name": "Team Average",
                "summary": summarize_goalie_history([], away_team_history, game_date, 0, away_team_priors),
                "start_probability": 1.0,
            }
        ]

    if home_goalie_candidates:
        home_goalies = [
            {
                "goalie_id": candidate["goalie_id"],
                "goalie_name": candidate["goalie_name"],
                "summary": build_goalie_summary(candidate, home_team, 1, home_team_history),
                "start_probability": float(candidate["start_probability"]),
            }
            for candidate in home_goalie_candidates
        ]
    else:
        home_goalies = [
            {
                "goalie_id": None,
                "goalie_name": "Team Average",
                "summary": summarize_goalie_history([], home_team_history, game_date, 1, home_team_priors),
                "start_probability": 1.0,
            }
        ]

    pair_results = []
    for away_goalie in away_goalies[:2]:
        for home_goalie in home_goalies[:2]:
            pair_probability = away_goalie["start_probability"] * home_goalie["start_probability"]
            feature_row = build_matchup_feature_row(
                away_team=away_team,
                home_team=home_team,
                game_date=game_date,
                away_history=away_team_history,
                home_history=home_team_history,
                priors=priors,
                away_goalie_summary=away_goalie["summary"],
                home_goalie_summary=home_goalie["summary"],
                away_team_priors=away_team_priors,
                home_team_priors=home_team_priors,
            )

            numeric_values = [float(feature_row[column]) for column in artifact["numeric_feature_columns"]]
            away_team_one_hot = [1.0 if away_team == code else 0.0 for code in artifact["team_codes"]]
            home_team_one_hot = [1.0 if home_team == code else 0.0 for code in artifact["team_codes"]]
            X = np.array([numeric_values + away_team_one_hot + home_team_one_hot], dtype=float)
            direct_model = {
                "x_mean": np.array(artifact["x_mean"], dtype=float),
                "x_std": np.array(artifact["x_std"], dtype=float),
                "y_mean": float(artifact["direct_y_mean"]),
                "beta": np.array(artifact["direct_beta"], dtype=float),
            }
            goals_model = {
                "x_mean": np.array(artifact["x_mean"], dtype=float),
                "x_std": np.array(artifact["x_std"], dtype=float),
                "y_mean": float(artifact["goals_y_mean"]),
                "beta": np.array(artifact["goals_beta"], dtype=float),
            }
            shots_model = {
                "x_mean": np.array(artifact["x_mean"], dtype=float),
                "x_std": np.array(artifact["x_std"], dtype=float),
                "y_mean": float(artifact["shots_y_mean"]),
                "beta": np.array(artifact["shots_beta"], dtype=float),
            }

            direct_stack_fp = float(_predict_ridge(direct_model, X)[0])
            predicted_goals = max(0.0, float(_predict_ridge(goals_model, X)[0]))
            predicted_shots = max(0.0, float(_predict_ridge(shots_model, X)[0]))
            structural_stack_fp = float(
                _predict_stack_surface(artifact["stack_surface"], predicted_goals, predicted_shots)
            )
            weight = float(artifact["ensemble_weight_direct"])
            predicted_stack_fp = weight * direct_stack_fp + (1.0 - weight) * structural_stack_fp
            residual_std = max(float(artifact["residual_std"]), 0.75)
            prob_positive = 1.0 - _standard_normal_cdf((0.0 - predicted_stack_fp) / residual_std)
            prob_5_plus = 1.0 - _standard_normal_cdf((5.0 - predicted_stack_fp) / residual_std)

            pair_results.append(
                {
                    "away_goalie_name": away_goalie["goalie_name"],
                    "home_goalie_name": home_goalie["goalie_name"],
                    "pair_probability": pair_probability,
                    "predicted_stack_fp": predicted_stack_fp,
                    "predicted_total_goals": predicted_goals,
                    "predicted_total_shots": predicted_shots,
                    "prob_positive": prob_positive,
                    "prob_5_plus": prob_5_plus,
                    "feature_row": feature_row,
                }
            )

    pair_total = sum(pair["pair_probability"] for pair in pair_results) or 1.0
    for pair in pair_results:
        pair["pair_probability"] /= pair_total

    weighted_stack_fp = sum(pair["pair_probability"] * pair["predicted_stack_fp"] for pair in pair_results)
    weighted_goals = sum(pair["pair_probability"] * pair["predicted_total_goals"] for pair in pair_results)
    weighted_shots = sum(pair["pair_probability"] * pair["predicted_total_shots"] for pair in pair_results)
    weighted_prob_positive = sum(pair["pair_probability"] * pair["prob_positive"] for pair in pair_results)
    weighted_prob_5_plus = sum(pair["pair_probability"] * pair["prob_5_plus"] for pair in pair_results)
    top_pair = max(pair_results, key=lambda pair: pair["pair_probability"])

    return {
        "matchup": f"{away_team} vs {home_team}",
        "team_a": away_team,
        "team_b": home_team,
        "predicted_stack_fp": round(weighted_stack_fp, 2),
        "prob_positive": round(weighted_prob_positive, 3),
        "prob_5_plus": round(weighted_prob_5_plus, 3),
        "tier": classify_stack_tier(
            weighted_stack_fp,
            weighted_prob_positive,
            weighted_prob_5_plus,
            tier_thresholds=artifact.get("tier_thresholds"),
        ),
        "predicted_total_goals": round(weighted_goals, 2),
        "predicted_total_shots": round(weighted_shots, 2),
        "season_expected_goals": round(top_pair["feature_row"]["season_expected_goals"], 2),
        "season_expected_shots": round(top_pair["feature_row"]["season_expected_shots"], 2),
        "form10_expected_goals": round(top_pair["feature_row"]["form10_expected_goals"], 2),
        "form10_expected_shots": round(top_pair["feature_row"]["form10_expected_shots"], 2),
        "rest_diff": round(top_pair["feature_row"]["rest_diff"], 1),
        "games_sample_floor": int(top_pair["feature_row"]["gp_min"]),
        "likely_away_goalie": away_goalies[0]["goalie_name"],
        "likely_away_goalie_prob": round(away_goalies[0]["start_probability"], 3),
        "likely_home_goalie": home_goalies[0]["goalie_name"],
        "likely_home_goalie_prob": round(home_goalies[0]["start_probability"], 3),
        "top_goalie_pair": f"{top_pair['away_goalie_name']} + {top_pair['home_goalie_name']}",
        "top_goalie_pair_prob": round(top_pair["pair_probability"], 3),
        "pair_results": [
            {
                "pair": f"{pair['away_goalie_name']} + {pair['home_goalie_name']}",
                "pair_probability": round(pair["pair_probability"], 3),
                "predicted_stack_fp": round(pair["predicted_stack_fp"], 2),
            }
            for pair in sorted(pair_results, key=lambda row: row["pair_probability"], reverse=True)
        ],
    }
