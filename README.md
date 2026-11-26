# NFL Play-Call Predictor

## Project Goal

The NFL Play-Call Predictor uses historical play-by-play data to predict whether an offense will run or pass in a given game situation. The project explores how predictable NFL play-calling can be using basic pre-snap information.

## MVP Scope

The minimum viable product will:

- Load one season of NFL play-by-play data.
- Keep valid run and pass plays.
- Use down, yards to go, and field position as input features.
- Train a logistic regression model.
- Compare the model's accuracy against a simple baseline.
- Produce run and pass probabilities for a game situation.

## Future Plans

Future versions may:

- Add score difference and time remaining.
- Include formation, offensive personnel, shotgun, and no-huddle information.
- Train and tune a gradient boosting model.
- Compare predictability across NFL offenses.
- Add charts or an interactive interface.
- Support multiple seasons of data.

## Tech Stack

- **Language:** Python
- **Data source:** NFL play-by-play data loaded with `nflreadpy`
- **Data processing:** pandas
- **Machine learning:** scikit-learn
- **Data storage:** Local CSV or Parquet files
- **Initial model:** Logistic regression
- **Development tools:** Git and GitHub

## Project Structure

```text
src/        Python source code
tests/      Project tests
data/       Local data files
docs/       Planning documents and diagrams




## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/shreyas-prasad-848/nfl-play-call-predictor.git
cd nfl-play-call-predictor
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and inspect the NFL play-by-play dataset:

```bash
python src/data_loader.py
```

The project downloads the 2025 NFL season and caches
the dataset locally as a Parquet file. An internet
connection is required for the initial download.

## Roadmap

### MVP
- [x] Set up project dependencies
- [x] Download and cache NFL play-by-play data
- [x] Clean and filter valid run/pass plays
- [x] Prepare features and target labels
- [ ] Split training and testing data
- [ ] Train a logistic regression model
- [ ] Evaluate accuracy against a baseline
- [ ] Implement interactive terminal predictions
- [ ] Test edge cases and error handling

### Future Features
- [ ] Rank NFL offenses by predictability
- [ ] Add additional pre-snap features
- [ ] Implement gradient boosting
- [ ] Tune model hyperparameters
- [ ] Support multiple seasons




### Data Preprocessing

To clean and inspect the downloaded dataset, run:

```bash
python src/preprocessing.py
```

The preprocessing pipeline:
- Retains only run and pass plays.
- Selects relevant features and identifiers.
- Removes missing values and invalid game situations.
- Encodes runs as 0 and passes as 1.
- Separates input features from target labels.

The three MVP model features are down, yards to go,
and distance from the opponent's end zone.




### Dataset Summary

After preprocessing the 2025 NFL play-by-play dataset:

| Metric | Value |
|---|---:|
| Original rows | 48,771 |
| Valid run/pass plays | 34,502 |
| Removed rows | 14,269 |
| Pass plays | 19,639 (56.92%) |
| Run plays | 14,863 (43.08%) |
| Missing values remaining | 0 |
| Model input features | 3 |