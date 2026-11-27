from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from data_loader import load_data
from preprocessing import clean_data, prepare_features
from model import split_data, train_baseline, train_model


def evaluate_models(baseline, model, X_test, y_test):
    baseline_predictions = baseline.predict(X_test)
    model_predictions = model.predict(X_test)

    baseline_accuracy = accuracy_score(
        y_test, baseline_predictions
    )
    model_accuracy = accuracy_score(
        y_test, model_predictions
    )

    improvement = model_accuracy - baseline_accuracy

    print("\nTesting accuracy:")
    print(f"Baseline: {baseline_accuracy:.2%}")
    print(f"Logistic regression: {model_accuracy:.2%}")
    print(f"Improvement: {improvement * 100:+.2f} percentage points")

    # Fix the label order so rows and columns always mean run, then pass.
    matrix = confusion_matrix(
        y_test, model_predictions, labels=[0, 1]
    )

    print("\nConfusion matrix:")
    print("                 Predicted Run  Predicted Pass")
    print(f"Actual Run       {matrix[0][0]:13d}  {matrix[0][1]:14d}")
    print(f"Actual Pass      {matrix[1][0]:13d}  {matrix[1][1]:14d}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        model_predictions,
        labels=[0, 1],
        target_names=["Run", "Pass"],
        zero_division=0
    ))


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

        print("\nTraining models...")
        baseline = train_baseline(X_train, y_train)
        model = train_model(X_train, y_train)

        evaluate_models(baseline, model, X_test, y_test)

    except Exception as error:
        print(f"Model evaluation failed: {error}")
        raise SystemExit(1)