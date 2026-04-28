from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from stack_predictive_model import (
    PredictiveStackModelError,
    build_histories_through_date,
    evaluate_matchup_with_model,
    extract_goalie_candidates,
    fetch_gamecenter_landing,
    infer_goalie_start_probabilities,
    infer_target_season,
    load_master_games,
    load_model_artifact,
    update_predictive_stack_model,
)
from update_goalie_stack_data import update_master_csv
from update_team_data import TeamDataUpdateError, update_team_data_csv


SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date_str}"
DEFAULT_TEAM_DATA_PATH = Path(__file__).with_name("team_data.csv")


class ScheduleFetchError(RuntimeError):
    pass


def fetch_schedule_json(date_str, timeout=10):
    try:
        response = requests.get(SCHEDULE_URL.format(date_str=date_str), timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ScheduleFetchError(f"Could not fetch NHL schedule for {date_str}: {exc}") from exc
    return response.json()


def get_todays_matchups(date_str):
    data = fetch_schedule_json(date_str)
    matchups = []

    for game_day in data.get("gameWeek", []):
        if game_day.get("date") != date_str:
            continue

        for game in game_day.get("games", []):
            away_team = game.get("awayTeam", {}).get("abbrev")
            home_team = game.get("homeTeam", {}).get("abbrev")

            if away_team and home_team:
                matchups.append(
                    {
                        "game_id": game.get("id"),
                        "start_time_utc": game.get("startTimeUTC"),
                        "away_team_id": game.get("awayTeam", {}).get("id"),
                        "home_team_id": game.get("homeTeam", {}).get("id"),
                        "away_team": away_team,
                        "home_team": home_team,
                    }
                )

    return matchups


def evaluate_todays_schedule(date_str=None, team_data_path=DEFAULT_TEAM_DATA_PATH):
    target_date = date_str or datetime.now().strftime("%Y-%m-%d")
    update_master_csv()
    update_predictive_stack_model()
    team_data_warning = None
    try:
        update_team_data_csv(team_data_path)
    except TeamDataUpdateError as exc:
        team_data_warning = str(exc)
    games_df = load_master_games()
    artifact = load_model_artifact()
    season = infer_target_season(games_df, target_date)
    team_histories, goalie_histories = build_histories_through_date(games_df, target_date)
    evaluated_games = []
    skipped_games = []

    for game in get_todays_matchups(target_date):
        away_team = game["away_team"]
        home_team = game["home_team"]

        try:
            preview = fetch_gamecenter_landing(game["game_id"])
            away_candidates = extract_goalie_candidates(preview, away_team, game["away_team_id"])
            home_candidates = extract_goalie_candidates(preview, home_team, game["home_team_id"])
            away_goalies = infer_goalie_start_probabilities(
                season=season,
                team_code=away_team,
                team_history=team_histories[(season, away_team)],
                goalie_histories=goalie_histories,
                candidates=away_candidates,
                game_date=pd.Timestamp(target_date).normalize(),
                current_is_home=0,
                priors=artifact["priors"],
            )
            home_goalies = infer_goalie_start_probabilities(
                season=season,
                team_code=home_team,
                team_history=team_histories[(season, home_team)],
                goalie_histories=goalie_histories,
                candidates=home_candidates,
                game_date=pd.Timestamp(target_date).normalize(),
                current_is_home=1,
                priors=artifact["priors"],
            )

            result = evaluate_matchup_with_model(
                away_team,
                home_team,
                target_date,
                games_df,
                artifact,
                away_goalie_candidates=away_goalies,
                home_goalie_candidates=home_goalies,
            )
        except Exception as exc:
            skipped_games.append(
                {
                    "matchup": f"{away_team} vs {home_team}",
                    "reason": str(exc),
                }
            )
            continue

        result["game_id"] = game["game_id"]
        result["start_time_utc"] = game["start_time_utc"]
        evaluated_games.append(result)

    evaluated_games.sort(key=lambda row: row["predicted_stack_fp"], reverse=True)

    return {
        "date": target_date,
        "evaluated_games": evaluated_games,
        "skipped_games": skipped_games,
        "model_info": {
            "training_rows": artifact["training_rows"],
            "ensemble_weight_direct": round(artifact["ensemble_weight_direct"], 2),
            "direct_alpha": artifact["direct_alpha"],
            "goals_alpha": artifact["goals_alpha"],
            "shots_alpha": artifact["shots_alpha"],
            "validation_mae": round(artifact["validation_metrics"]["mae"], 3),
            "validation_rmse": round(artifact["validation_metrics"]["rmse"], 3),
            "validation_spearman": round(artifact["validation_metrics"]["spearman"], 3),
            "latest_training_date": artifact["latest_training_date"],
        },
        "team_data_warning": team_data_warning,
    }


def print_ranked_report(report):
    print(f"\nTODAY'S NHL STACK CANDIDATES: {report['date']}")
    print("----------------------------------------")
    model_info = report["model_info"]
    print(
        f"Model: ensemble ridge | games={model_info['training_rows']} | "
        f"w_direct={model_info['ensemble_weight_direct']:.2f} | "
        f"alphas=({model_info['direct_alpha']}, {model_info['goals_alpha']}, {model_info['shots_alpha']}) | "
        f"val_mae={model_info['validation_mae']:.3f} | "
        f"val_rmse={model_info['validation_rmse']:.3f} | "
        f"val_spearman={model_info['validation_spearman']:.3f} | "
        f"through={model_info['latest_training_date']}"
    )
    print("")

    if report["team_data_warning"]:
        print(f"team_data refresh warning: {report['team_data_warning']}")
        print("")

    if not report["evaluated_games"]:
        print("No scheduled games could be evaluated with the current team_data.csv coverage.")
    else:
        for idx, game in enumerate(report["evaluated_games"], start=1):
            print(
                f"{idx}. {game['matchup']} | "
                f"predicted_stack_fp={game['predicted_stack_fp']:.2f} | "
                f"p5+={game['prob_5_plus']:.1%} | "
                f"p>0={game['prob_positive']:.1%} | "
                f"tier={game['tier']} | "
                f"goalies={game['likely_away_goalie']} ({game['likely_away_goalie_prob']:.0%}) / "
                f"{game['likely_home_goalie']} ({game['likely_home_goalie_prob']:.0%}) | "
                f"pred_goals={game['predicted_total_goals']:.2f} | "
                f"pred_shots={game['predicted_total_shots']:.2f}"
            )

    if report["skipped_games"]:
        print("\nSKIPPED MATCHUPS")
        print("----------------")
        for game in report["skipped_games"]:
            print(f"{game['matchup']} | {game['reason']}")


if __name__ == "__main__":
    try:
        print_ranked_report(evaluate_todays_schedule())
    except (ScheduleFetchError, TeamDataUpdateError, PredictiveStackModelError) as exc:
        print(exc)
