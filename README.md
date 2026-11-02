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
