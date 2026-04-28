# NHL Goalie Stack Model

## Overview

This project builds a predictive model to identify optimal goalie “stack” combinations in fantasy hockey (two goalies playing against each other).

The system pulls NHL data, builds a historical dataset, trains a predictive model, and ranks daily matchups based on expected fantasy performance and probability outcomes.

## What It Does

* Collects historical NHL game data via API
* Computes fantasy points for each goalie
* Builds a dataset of goalie matchups (~4,000+ games)
* Trains an ensemble predictive model
* Evaluates daily matchups and ranks goalie pairs

Output includes:

* predicted fantasy points (`predicted_stack_fp`)
* probability of positive outcome (`p > 0`)
* probability of strong performance (`p ≥ 5`)
* matchup tier classification

Example output:

```
1. BOS vs BUF | predicted_stack_fp=3.51 | p5+=39.2% | p>0=74.1% | tier=STRONG STACK
2. MIN vs DAL | predicted_stack_fp=2.92 | p5+=35.1% | p>0=70.4% | tier=STRONG STACK
```

## Model

The model uses an ensemble approach:

* Ridge regression to predict fantasy points directly
* Ridge regression to predict goals and shots
* A nonlinear surface model to convert goals/shots → fantasy points
* Final predictions are a weighted blend of both approaches

The model also estimates uncertainty:

* probability of scoring > 0
* probability of scoring ≥ 5

## Data

Data is collected from the NHL API:

* game schedules and box scores
* goalie stats (saves, shots, goals allowed)
* team-level statistics

A master dataset is maintained and updated automatically.

## Project Structure

```
├── main.py
├── scripts/
│   ├── stack_predictive_model.py
│   ├── update_goalie_stack_data.py
│   └── update_team_data.py
├── data/
│   ├── goalie_stack_games_master.csv
│   └── team_data.csv
├── stack_predictive_model.json
```

## How to Run

Install dependencies:

```
pip install pandas numpy requests
```

Run:

```
python main.py
```

This will:

1. Update historical game data
2. Update team stats
3. Train/update the model
4. Output ranked goalie stack candidates for today

## Key Features

* End-to-end data pipeline (API → dataset → model → predictions)
* Feature engineering using rolling windows and exponential weighting
* Ensemble modeling approach
* Probabilistic outputs (not just point estimates)
* Fully automated daily update workflow

## Notes

This project was built as a personal analytics tool and demonstrates applied machine learning, data engineering, and API integration in a sports context.
