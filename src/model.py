from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from data_loader import load_data
from preprocessing import clean_data, prepare_features, FEATURES


def split_data(data, train_ratio=0.8):
    if not 0 < train_ratio < 1:
        raise ValueError("Training ratio must be between 0 and 1.")

    weeks = sorted(data["week"].unique())
    split_index = int(len(weeks) * train_ratio)

    if split_index == 0 or split_index == len(weeks):
        raise ValueError("Not enough weeks to split the dataset.")

    # Keep entire weeks together to simulate predicting future games.
    train_weeks = weeks[:split_index]
    test_weeks = weeks[split_index:]

    train_data = data[data["week"].isin(train_weeks)].copy()
    test_data = data[data["week"].isin(test_weeks)].copy()

    return train_data, test_data


def train_baseline(X_train, y_train):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    return baseline


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":
    try:
        raw_data = load_data()
        cleaned_data = clean_data(raw_data)

        train_data, test_data = split_data(cleaned_data)

        X_train, y_train = prepare_features(train_data)
        X_test, y_test = prepare_features(test_data)

        print("\nDataset split:")
        print(f"Training plays: {len(train_data)}")
        print(f"Testing plays: {len(test_data)}")

        print(f"Training weeks: {sorted(train_data['week'].unique().tolist())}")
        print(f"Testing weeks: {sorted(test_data['week'].unique().tolist())}")

        print("\nTraining baseline...")
        baseline = train_baseline(X_train, y_train)

        baseline_label = baseline.predict(X_train.iloc[:1])[0]
        baseline_play = "Pass" if baseline_label == 1 else "Run"

        print(f"Baseline predicts: {baseline_play}")

        print("\nTraining logistic regression...")
        model = train_model(X_train, y_train)

        print("Model trained successfully!")

        print("\nLearned coefficients:")
        for feature, coefficient in zip(FEATURES, model.coef_[0]):
            print(f"{feature}: {coefficient:.4f}")

        print(f"Intercept: {model.intercept_[0]:.4f}")

        print("\nTraining accuracy:")
        print(f"Baseline: {baseline.score(X_train, y_train):.2%}")
        print(f"Logistic regression: {model.score(X_train, y_train):.2%}")

    except Exception as error:
        print(f"Model training failed: {error}")
        raise SystemExit(1)