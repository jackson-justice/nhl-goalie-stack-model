"""Pregame betting-market lines (game total and moneyline) from ESPN's public odds feed.

The market total is the best available estimate of how many goals a game will have, which is the
part of a goalie stack the box-score features can't predict on their own.
"""

import math
import statistics
import time
from pathlib import Path

import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ODDS_FILE = DATA_DIR / "game_odds.csv"
SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
ODDS_URL = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events/{event_id}/competitions/{event_id}/odds"
ODDS_COLUMNS = [
    "game_id",
    "date",
    "away_team",
    "home_team",
    "espn_event_id",
    "total_line",
    "market_total_goals",
    "market_home_win_prob",
    "n_total_books",
    "n_ml_books",
]
# ESPN abbreviations that differ from the NHL API's.
ESPN_TO_NHL = {"TB": "TBL", "NJ": "NJD", "SJ": "SJS", "LA": "LAK", "UTAH": "UTA", "WAS": "WSH", "MON": "MTL"}


def _get_json(url, params=None, timeout=20, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt < max_retries - 1:
                time.sleep(1.0 + attempt)
    return None


def _decimal_odds(american):
    american = float(american)
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / -american)


def _devig(odds_a, odds_b):
    """Fair probability of side a from a two-way American-odds market."""
    raw_a = 1.0 / _decimal_odds(odds_a)
    raw_b = 1.0 / _decimal_odds(odds_b)
    return raw_a / (raw_a + raw_b)


def _poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, int(k) + 1):
        term *= lam / i
        total += term
    return total


def _prob_over(line, lam):
    """P(over) for a total line, with pushes on whole-number lines removed."""
    if float(line).is_integer():
        over = 1.0 - _poisson_cdf(line, lam)
        under = _poisson_cdf(line - 1, lam)
        return over / (over + under)
    return 1.0 - _poisson_cdf(math.floor(line), lam)


def implied_total_goals(line, prob_over):
    """Expected total goals that makes a Poisson total match the market's over probability."""
    low, high = 1.0, 14.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        if _prob_over(line, mid) < prob_over:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def parse_odds_items(items):
    """Consensus market numbers from ESPN's per-book odds list (median across pregame books)."""
    totals, lines, home_probs = [], [], []
    for item in items:
        provider = (item.get("provider") or {}).get("name", "")
        if "live" in provider.lower():
            continue

        line, over, under = item.get("overUnder"), item.get("overOdds"), item.get("underOdds")
        if line is not None and over is not None and under is not None:
            try:
                prob_over = _devig(over, under)
                # Lines priced past -250 either way are usually alternate or stale lines.
                if 0.25 <= prob_over <= 0.75 and 4.0 <= float(line) <= 9.0:
                    totals.append(implied_total_goals(float(line), prob_over))
                    lines.append(float(line))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        home_ml = (item.get("homeTeamOdds") or {}).get("moneyLine")
        away_ml = (item.get("awayTeamOdds") or {}).get("moneyLine")
        if home_ml is not None and away_ml is not None:
            try:
                if abs(float(home_ml)) < 1000 and abs(float(away_ml)) < 1000:
                    home_probs.append(_devig(home_ml, away_ml))
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    return {
        "total_line": statistics.median(lines) if lines else None,
        "market_total_goals": statistics.median(totals) if totals else None,
        "market_home_win_prob": statistics.median(home_probs) if home_probs else None,
        "n_total_books": len(totals),
        "n_ml_books": len(home_probs),
    }


def fetch_espn_events(date_str):
    """ESPN event ids for a date (YYYY-MM-DD), keyed by (away, home) in NHL abbreviations."""
    data = _get_json(SCOREBOARD_URL, params={"dates": date_str.replace("-", "")})
    events = {}
    for event in (data or {}).get("events", []):
        competitors = event.get("competitions", [{}])[0].get("competitors", [])
        teams = {}
        for competitor in competitors:
            abbrev = competitor.get("team", {}).get("abbreviation", "")
            teams[competitor.get("homeAway")] = ESPN_TO_NHL.get(abbrev, abbrev)
        if "away" in teams and "home" in teams:
            events[(teams["away"], teams["home"])] = event["id"]
    return events


def fetch_event_odds(event_id):
    data = _get_json(ODDS_URL.format(event_id=event_id))
    return parse_odds_items((data or {}).get("items", []))


def odds_for_games(games, pause_seconds=0.05):
    """Market lines for games given as dicts with game_id, date (YYYY-MM-DD), away_team, home_team."""
    rows = []
    events_by_date = {}
    for game in games:
        date_str = game["date"]
        key = (game["away_team"], game["home_team"])
        event_id = None
        # ESPN files some late games under the next day, so look one day either side if needed.
        for offset in (0, -1, 1):
            lookup = (pd.Timestamp(date_str) + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            if lookup not in events_by_date:
                events_by_date[lookup] = fetch_espn_events(lookup)
                time.sleep(pause_seconds)
            event_id = events_by_date[lookup].get(key)
            if event_id:
                break

        row = {"game_id": game["game_id"], "date": date_str, "away_team": key[0], "home_team": key[1], "espn_event_id": event_id}
        row.update(fetch_event_odds(event_id) if event_id else parse_odds_items([]))
        rows.append(row)
        time.sleep(pause_seconds)
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def load_odds(odds_file=ODDS_FILE):
    path = Path(odds_file)
    if not path.exists():
        return pd.DataFrame(columns=ODDS_COLUMNS)
    return pd.read_csv(path)


def update_odds_csv(games_df, odds_file=ODDS_FILE, save_every=200):
    """Fetch market lines for any completed game in games_df that isn't in the odds file yet."""
    existing = load_odds(odds_file)
    have = set(existing["game_id"].astype(int)) if not existing.empty else set()
    todo = games_df[~games_df["game_id"].astype(int).isin(have)]
    if todo.empty:
        return existing

    print(f"Fetching market lines for {len(todo)} games...")
    frames = [existing] if not existing.empty else []
    records = [
        {
            "game_id": int(game.game_id),
            "date": pd.Timestamp(game.date).strftime("%Y-%m-%d"),
            "away_team": game.away_team,
            "home_team": game.home_team,
        }
        for game in todo.itertuples(index=False)
    ]
    for start in range(0, len(records), save_every):
        frames.append(odds_for_games(records[start : start + save_every]))
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="game_id", keep="last")
        combined.sort_values(["date", "game_id"]).to_csv(odds_file, index=False)
        print(f"  {min(start + save_every, len(records))}/{len(records)} games")
    return combined


if __name__ == "__main__":
    from stack_predictive_model import load_master_games

    odds = update_odds_csv(load_master_games())
    print(f"{len(odds)} games, {odds['market_total_goals'].notna().mean():.1%} with a market total.")
