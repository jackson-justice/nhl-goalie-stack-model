from pathlib import Path
import time

import pandas as pd
import requests


STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
CLUB_STATS_URL = "https://api-web.nhle.com/v1/club-stats/{team}/now"
OUTPUT_FILE = Path(__file__).with_name("team_data.csv")


class TeamDataUpdateError(RuntimeError):
    pass


def fetch_json(url, timeout=15):
    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

    raise TeamDataUpdateError(f"Request failed for {url}: {last_error}") from last_error


def get_standings():
    data = fetch_json(STANDINGS_URL)
    standings = data.get("standings", [])

    if not standings:
        raise TeamDataUpdateError("Standings response did not include any teams.")

    return standings


def get_team_club_stats(team_code):
    return fetch_json(CLUB_STATS_URL.format(team=team_code))


def compute_shots_per_game(team_code, games_played):
    club_stats = get_team_club_stats(team_code)
    total_shots_for = sum(player.get("shots", 0) for player in club_stats.get("skaters", []))
    total_shots_against = sum(goalie.get("shotsAgainst", 0) for goalie in club_stats.get("goalies", []))

    if games_played <= 0:
        return 0.0, 0.0

    return round(total_shots_for / games_played, 1), round(total_shots_against / games_played, 1)


def build_team_data_rows(pause_seconds=0.1):
    rows = []

    for standing in get_standings():
        team_code = standing["teamAbbrev"]["default"]
        games_played = int(standing["gamesPlayed"])
        sog_pg, sa_pg = compute_shots_per_game(team_code, games_played)

        rows.append(
            {
                "team": team_code,
                "gf_total": int(standing["goalFor"]),
                "ga_total": int(standing["goalAgainst"]),
                "sog_pg": sog_pg,
                "sa_pg": sa_pg,
                "games_played": games_played,
            }
        )

        time.sleep(pause_seconds)

    return rows


def build_team_data_frame():
    rows = build_team_data_rows()
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def update_team_data_csv(output_path=OUTPUT_FILE):
    df = build_team_data_frame()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    try:
        df = update_team_data_csv()
        print(f"Updated {OUTPUT_FILE.name} with {len(df)} teams.")
        print(df.head(10).to_string(index=False))
    except TeamDataUpdateError as exc:
        print(exc)
