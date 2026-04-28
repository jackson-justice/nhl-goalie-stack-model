import requests
import pandas as pd
import time
from pathlib import Path


OUTPUT_FILE = "goalie_stack_games_master.csv"
GAME_COLUMNS = [
    "game_id",
    "date",
    "season",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "total_goals",
    "goal_diff",
    "overtime_flag",
    "away_goalie",
    "home_goalie",
    "away_goalie_id",
    "home_goalie_id",
    "away_saves",
    "home_saves",
    "away_shots_against",
    "home_shots_against",
    "total_shots",
    "away_fp",
    "home_fp",
    "stack_fp",
]


def goalie_fantasy_points(goalie):
    w = 1 if goalie.get("decision") == "W" else 0
    otl = 1 if goalie.get("decision") == "O" else 0
    ga = goalie.get("goalsAgainst", 0)
    sv = goalie.get("saves", 0)
    so = 1 if (ga == 0 and goalie.get("starter") is True and goalie.get("toi") != "00:00") else 0
    return 4 * w + 1 * otl - 2 * ga + 0.2 * sv + 3 * so


def get_starting_goalie(goalies):
    for goalie in goalies:
        if goalie.get("starter") is True:
            return goalie
    return None


def safe_get_json(url, timeout=10, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"    Request failed ({attempt + 1}/{max_retries}): {e}")
            time.sleep(1)
    return None


def get_games_for_date(date_str):
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    data = safe_get_json(url)
    if data is None:
        return []

    games = []
    for game_day in data.get("gameWeek", []):
        if game_day.get("date") == date_str:
            for game in game_day.get("games", []):
                if game.get("gameState") == "OFF":   # completed games only
                    games.append(game)
    return games


def parse_game(game_id):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    data = safe_get_json(url)
    if data is None:
        return None

    away_team = data["awayTeam"]["abbrev"]
    home_team = data["homeTeam"]["abbrev"]

    away_goalies = data["playerByGameStats"]["awayTeam"]["goalies"]
    home_goalies = data["playerByGameStats"]["homeTeam"]["goalies"]

    away_starter = get_starting_goalie(away_goalies)
    home_starter = get_starting_goalie(home_goalies)

    if away_starter is None or home_starter is None:
        return None

    away_score = data["awayTeam"].get("score", 0)
    home_score = data["homeTeam"].get("score", 0)

    away_fp = goalie_fantasy_points(away_starter)
    home_fp = goalie_fantasy_points(home_starter)

    period_type = data.get("gameOutcome", {}).get("lastPeriodType", "REG")
    overtime_flag = 1 if period_type in ["OT", "SO"] else 0

    return {
        "game_id": data.get("id"),
        "date": data.get("gameDate"),
        "season": data.get("season"),
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_score,
        "home_score": home_score,
        "total_goals": away_score + home_score,
        "goal_diff": abs(away_score - home_score),
        "overtime_flag": overtime_flag,
        "away_goalie": away_starter["name"]["default"],
        "home_goalie": home_starter["name"]["default"],
        "away_goalie_id": away_starter.get("playerId"),
        "home_goalie_id": home_starter.get("playerId"),
        "away_saves": away_starter.get("saves", 0),
        "home_saves": home_starter.get("saves", 0),
        "away_shots_against": away_starter.get("shotsAgainst", 0),
        "home_shots_against": home_starter.get("shotsAgainst", 0),
        "total_shots": away_starter.get("shotsAgainst", 0) + home_starter.get("shotsAgainst", 0),
        "away_fp": round(away_fp, 2),
        "home_fp": round(home_fp, 2),
        "stack_fp": round(away_fp + home_fp, 2),
    }


def collect_games(start_date, end_date, partial_save_every=100):
    dates = pd.date_range(start=start_date, end=end_date)
    rows = []

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"Checking {date_str}...")

        games = get_games_for_date(date_str)

        for game in games:
            game_id = game["id"]
            row = parse_game(game_id)

            if row is not None:
                rows.append(row)
                print(
                    f"  Added {row['away_team']} @ {row['home_team']} | "
                    f"{row['away_goalie']} vs {row['home_goalie']} | "
                    f"stack_fp={row['stack_fp']}"
                )

            if len(rows) > 0 and len(rows) % partial_save_every == 0:
                pd.DataFrame(rows).drop_duplicates(subset="game_id").to_csv(
                    "goalie_stack_games_partial.csv", index=False
                )
                print("  Partial save complete.")

            time.sleep(0.25)

    if not rows:
        return pd.DataFrame(columns=GAME_COLUMNS)

    df = pd.DataFrame(rows).drop_duplicates(subset="game_id").sort_values(["date", "game_id"])
    return df


if __name__ == "__main__":
    df = collect_games("2023-10-01", "2026-03-17")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} games to {OUTPUT_FILE}")
