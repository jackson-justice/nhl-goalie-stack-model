from datetime import datetime

import pandas as pd
import requests

from stack_predictive_model import (
    PredictiveStackModelError,
    build_histories_through_date,
    compute_carryover_priors,
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
from update_odds_data import odds_for_games, update_odds_csv


SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date_str}"


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


def fetch_market_lines(target_date, matchups):
    """Today's betting lines keyed by (away, home)."""
    games = [
        {"game_id": game["game_id"], "date": target_date, "away_team": game["away_team"], "home_team": game["home_team"]}
        for game in matchups
    ]
    lines = odds_for_games(games)
    return {
        (row.away_team, row.home_team): {
            "market_total_goals": row.market_total_goals,
            "market_home_win_prob": row.market_home_win_prob,
        }
        for row in lines.itertuples(index=False)
    }


def evaluate_todays_schedule(date_str=None):
    target_date = date_str or datetime.now().strftime("%Y-%m-%d")
    warnings = []
    update_master_csv()
    try:
        update_odds_csv(load_master_games())
    except Exception as exc:
        warnings.append(f"Could not refresh historical betting lines: {exc}")
    update_predictive_stack_model()
    games_df = load_master_games()
    artifact = load_model_artifact()
    season = infer_target_season(games_df, target_date)
    team_histories, goalie_histories = build_histories_through_date(games_df, target_date)
    carryover = compute_carryover_priors(games_df, artifact["priors"], extra_seasons=[season])
    evaluated_games = []
    skipped_games = []
    matchups = get_todays_matchups(target_date)
    try:
        market_lines = fetch_market_lines(target_date, matchups)
    except Exception as exc:
        market_lines = {}
        warnings.append(f"Could not fetch today's betting lines: {exc}")

    for game in matchups:
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
                carryover=carryover,
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
                carryover=carryover,
            )

            result = evaluate_matchup_with_model(
                away_team,
                home_team,
                target_date,
                games_df,
                artifact,
                away_goalie_candidates=away_goalies,
                home_goalie_candidates=home_goalies,
                carryover=carryover,
                market=market_lines.get((away_team, home_team)),
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
    missing_lines = [game["matchup"] for game in evaluated_games if not game["has_market_line"]]
    if missing_lines:
        warnings.append(f"No betting line yet for {', '.join(missing_lines)}; those use team stats only.")

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
            "baseline_mae": round(artifact["validation_metrics"].get("baseline_mae", float("nan")), 3),
            "validation_rmse": round(artifact["validation_metrics"]["rmse"], 3),
            "validation_spearman": round(artifact["validation_metrics"]["spearman"], 3),
            "latest_training_date": artifact["latest_training_date"],
            "stack_top_fifth_fp": artifact["validation_metrics"].get("top_fifth_fp"),
            "stack_bottom_fifth_fp": artifact["validation_metrics"].get("bottom_fifth_fp"),
            "goalie_metrics": artifact.get("goalie_model", {}).get("validation_metrics"),
        },
        "warnings": warnings,
    }


def goalie_rankings(report):
    """One row per team tonight: projected FP for its starter, with the likely starter named."""
    rows = []
    for game in report["evaluated_games"]:
        teams = {"away": game["team_a"], "home": game["team_b"]}
        for side, opponent in (("away", "home"), ("home", "away")):
            projection = game["goalie_projections"].get(side)
            if not projection:
                continue
            rows.append(
                {
                    "team": teams[side],
                    "opponent": teams[opponent],
                    "venue": "@" if side == "away" else "vs",
                    "goalie": game[f"likely_{side}_goalie"],
                    "start_prob": game[f"likely_{side}_goalie_prob"],
                    **projection,
                }
            )
    return sorted(rows, key=lambda row: row["predicted_fp"], reverse=True)


def print_ranked_report(report):
    model_info = report["model_info"]
    print(f"\nNHL GOALIE PROJECTIONS: {report['date']}")
    print("----------------------------------------")
    goalie_metrics = model_info["goalie_metrics"]
    if goalie_metrics:
        print(
            f"Backtest: top fifth of projections averaged {goalie_metrics['top_fifth_fp']:.2f} FP vs "
            f"{goalie_metrics['bottom_fifth_fp']:.2f} for the bottom fifth | "
            f"mae={goalie_metrics['mae']:.3f} (avg-guess {goalie_metrics['baseline_mae']:.3f}) | "
            f"through={model_info['latest_training_date']}"
        )
    print("Projections assume the goalie starts; check confirmed starters before lock.")
    print("")

    for warning in report["warnings"]:
        print(f"warning: {warning}")
    if report["warnings"]:
        print("")

    if not report["evaluated_games"]:
        print("No scheduled games could be evaluated.")
    for idx, row in enumerate(goalie_rankings(report), start=1):
        print(
            f"{idx}. {row['goalie']} ({row['team']} {row['venue']} {row['opponent']}) | "
            f"projected_fp={row['predicted_fp']:.2f} | p5+={row['prob_5_plus']:.1%} | "
            f"p>0={row['prob_positive']:.1%} | win_prob={row['win_prob']:.0%} | "
            f"exp_shots_against={row['expected_shots_against']:.1f} | start_prob={row['start_prob']:.0%}"
        )

    print(f"\nGOALIE STACKS (both goalies in one game): {report['date']}")
    print("----------------------------------------")
    print(
        f"Model: ensemble ridge | games={model_info['training_rows']} | "
        f"w_direct={model_info['ensemble_weight_direct']:.2f} | "
        f"alphas=({model_info['direct_alpha']}, {model_info['goals_alpha']}, {model_info['shots_alpha']}) | "
        f"val_mae={model_info['validation_mae']:.3f} (avg-guess {model_info['baseline_mae']:.3f}) | "
        f"val_spearman={model_info['validation_spearman']:.3f}"
    )
    if model_info["stack_top_fifth_fp"] is not None:
        print(
            f"Backtest: top fifth of games averaged {model_info['stack_top_fifth_fp']:.2f} FP vs "
            f"{model_info['stack_bottom_fifth_fp']:.2f} for the bottom fifth."
        )
    print("Stack results are mostly luck; treat these tiers as a slight tilt, not a pick.")
    print("")

    for idx, game in enumerate(report["evaluated_games"], start=1):
        print(
            f"{idx}. {game['matchup']} | "
            f"predicted_stack_fp={game['predicted_stack_fp']:.2f} | "
            f"p5+={game['prob_5_plus']:.1%} | "
            f"p>0={game['prob_positive']:.1%} | "
            f"tier={game['tier']} | "
            f"market_total={game['market_total_goals']:.2f} | "
            f"goalies={game['likely_away_goalie']} ({game['likely_away_goalie_prob']:.0%}) / "
            f"{game['likely_home_goalie']} ({game['likely_home_goalie_prob']:.0%})"
        )

    if report["skipped_games"]:
        print("\nSKIPPED MATCHUPS")
        print("----------------")
        for game in report["skipped_games"]:
            print(f"{game['matchup']} | {game['reason']}")


if __name__ == "__main__":
    try:
        print_ranked_report(evaluate_todays_schedule())
    except (ScheduleFetchError, PredictiveStackModelError) as exc:
        print(exc)
