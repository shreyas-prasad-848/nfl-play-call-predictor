# NFL Play-Call Predictor — Eisenhower Matrix

## Do — MVP

- Load one NFL season and keep valid run and pass plays.
- Train a logistic regression model using down, yards to go, and field position.
- Display model accuracy and predicted run/pass probabilities.

## Schedule — Version 2.0+

- Add score difference, time remaining, formation, personnel, shotgun, and no-huddle features.
- Use a gradient boosting model and tune its settings.
- Rank NFL offenses from most predictable to least predictable.

## Delegate — Use a Library

- Use `nflreadpy` to obtain play-by-play data.
- Use `pandas` to filter and prepare the data.
- Use `scikit-learn` to split the data, train the model, and calculate accuracy.

## Delete — Outside the MVP

- No web or mobile interface.
- No live in-game predictions.
- No weather, injury, player identity, defensive tendency, or advanced play-category features.
