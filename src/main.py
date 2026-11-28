from data_loader import load_data
from preprocessing import clean_data, prepare_features
from model import (
    split_data,
    train_baseline,
    train_model,
    predict_play
)
from evaluation import evaluate_models


def prediction_loop(model):
    print("\n--- Interactive Play-Call Predictor ---")
    print("Enter a game situation to predict run or pass.")
    print("Enter 'q' at any prompt to quit.")

    while True:
        try:
            down = input("\nDown (1-4): ").strip()

            if down.lower() == "q":
                break

            down = int(down)

            yards_to_go = input("Yards to go: ").strip()

            if yards_to_go.lower() == "q":
                break

            yards_to_go = int(yards_to_go)

            field_position = input(
                "Yards from opponent's end zone (1-99): "
            ).strip()

            if field_position.lower() == "q":
                break

            field_position = int(field_position)

            result = predict_play(
                model, down, yards_to_go, field_position
            )

            print("\n--- Prediction Results ---")
            print(f"Down: {down}")
            print(f"Yards to go: {yards_to_go}")
            print(f"Field position: {field_position}")

            print(f"\nRun probability: {result['run_probability']:.2%}")
            print(f"Pass probability: {result['pass_probability']:.2%}")
            print(f"Predicted play: {result['predicted_play'].upper()}")

        except ValueError as error:
            print(f"\nInvalid input: {error}")
            print("Please enter a valid game situation.")

    print("\nThank you for using the NFL Play-Call Predictor!")


def main():
    print("=== NFL Play-Call Predictor ===")

    raw_data = load_data()
    cleaned_data = clean_data(raw_data)

    print(f"\nValid regular-season plays: {len(cleaned_data):,}")

    train_data, test_data = split_data(cleaned_data)

    X_train, y_train = prepare_features(train_data)
    X_test, y_test = prepare_features(test_data)

    print(f"Training plays: {len(train_data):,}")
    print(f"Testing plays: {len(test_data):,}")

    print("\nTraining models...")
    baseline = train_baseline(X_train, y_train)
    model = train_model(X_train, y_train)

    evaluate_models(baseline, model, X_test, y_test)

    prediction_loop(model)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting NFL Play-Call Predictor.")
    except Exception as error:
        print(f"\nUnable to run the predictor: {error}")
        raise SystemExit(1)