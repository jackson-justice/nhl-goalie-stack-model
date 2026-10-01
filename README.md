# NHL Goalie Stack Model

## Overview

This project projects fantasy points for NHL goalies. It ranks tonight's individual starters and goalie "stack" combinations (two goalies playing against each other).

The system pulls NHL data and pregame betting lines, builds a historical dataset, trains predictive models, and ranks tonight's goalies and matchups by expected fantasy points and probability outcomes.

## What It Does

* Collects historical NHL game data via API
* Computes fantasy points for each goalie
* Builds a dataset of goalie matchups (~4,000+ games)
* Trains a single-goalie model and an ensemble stack model
* Estimates who will start in goal for each team
* Ranks tonight's goalies and goalie pairs

Output includes:

* projected fantasy points per goalie (`projected_fp`) and per stack (`predicted_stack_fp`)
* probability of positive outcome (`p > 0`)
* probability of strong performance (`p ≥ 5`)
* likely starter and start probability
* matchup tier classification for stacks

Example output:

```
NHL GOALIE PROJECTIONS: 2026-10-01
1. K. Vejmelka (UTA vs CHI) | projected_fp=2.40 | p5+=31.5% | p>0=64.7% | win_prob=66% | exp_shots_against=26.0 | start_prob=98%
2. T. Jarry (EDM @ VAN) | projected_fp=2.35 | p5+=31.0% | p>0=64.4% | win_prob=66% | exp_shots_against=26.1 | start_prob=94%

GOALIE STACKS (both goalies in one game): 2026-10-01
1. MIN vs NSH | predicted_stack_fp=2.99 | p5+=35.3% | p>0=71.2% | tier=STRONG STACK | market_total=6.40 | goalies=J. Wallstedt (82%) / J. Saros (81%)
```

## Models

**Single goalie.** Ridge regression on the betting market's win probability and expected total goals, plus expected shots against. The win is worth 4 points, so the moneyline carries most of the signal. Goalie stats such as save percentage added nothing once the market was included, because closing lines already price in the starter. Probabilities come from the actual spread of out-of-sample errors.

**Stack.** An ensemble of ridge regression on stack fantasy points directly, and ridge models for total goals and shots fed through a goals/shots → fantasy points surface. Features are shot volume plus the market total and favorite strength. Probabilities assume normally distributed errors.

**Starting goalie.** A softmax over each candidate's share of the team's starts this season and in the last 5 games, with back-to-back adjustments. Weights were fit on 2024-25 and 2025-26 starts. Early in the season, last season's games played seed the shares.

### How well it predicts

Held-out results (trained only on earlier seasons):

| | 2024-25 | 2025-26 |
|---|---|---|
| Single goalie: top fifth vs bottom fifth of projections | 2.96 vs 0.56 FP | 2.01 vs 0.86 FP |
| Stack: top fifth vs bottom fifth of games | no edge | 3.75 vs 2.36 FP |
| Starting goalie: top pick correct | 68% | 68% |

Stack results are mostly luck. A stack's score is driven by total goals, which even the betting market barely predicts. Games expected to be high-scoring also bring more shots, and the extra saves cancel out the extra goals. Treat stack tiers as a slight tilt. The single-goalie projections are the more useful output, but they don't know confirmed starters, so check those before lineups lock.

## Data

Data is collected from the NHL API:

* game schedules and box scores
* goalie stats (saves, shots, goals allowed)
* team-level statistics

Pregame betting lines (game total and moneyline) come from ESPN's public odds feed via `scripts/update_odds_data.py` and are stored in `data/game_odds.csv`. Each line is converted to implied expected total goals and a de-vigged home win probability, taking the median across books. This feed is unofficial. If it's unavailable, or a game has no line yet, the models fall back to team stats and the report prints a warning.

A master dataset is maintained and updated automatically.

## Project Structure

```
├── scripts/
│   ├── main.py
│   ├── stack_predictive_model.py
│   ├── update_goalie_stack_data.py
│   ├── update_odds_data.py
│   └── update_team_data.py
├── data/
│   ├── goalie_stack_games_master.csv
│   ├── game_odds.csv
│   ├── team_data.csv
│   └── stack_predictive_model.json
```

## How to Run

Install dependencies:

```
pip install pandas numpy requests
```

Run:

```
python scripts/main.py
```

This will:

1. Update historical game data and betting lines
2. Train/update the models
3. Fetch tonight's betting lines and likely starters
4. Output ranked goalies and goalie stacks for today

Run it close to puck drop so lines and starters are as current as possible.

`scripts/update_team_data.py` still exists but is not part of the daily run, because the models don't use `team_data.csv`.

## Key Features

* End-to-end data pipeline (API → dataset → model → predictions)
* Feature engineering using rolling windows and exponential weighting
* Ensemble modeling approach
* Probabilistic outputs (not just point estimates)
* Fully automated daily update workflow

## Notes

This project was built as a personal analytics tool and demonstrates applied machine learning, data engineering, and API integration in a sports context.
