from data_loader import load_data


FEATURES = ["down", "ydstogo", "yardline_100"]

COLUMNS = [
    "game_id",
    "week",
    "posteam",
    "down",
    "ydstogo",
    "yardline_100",
    "play_type"
]


def clean_data(data):
    # Only keep plays relevant to binary classification.
    cleaned = data[data["play_type"].isin(["run", "pass"])]
    cleaned = cleaned[COLUMNS].copy()

    cleaned = cleaned.dropna()

    # Remove invalid game situations to avoid misleading the model.
    cleaned = cleaned[
        cleaned["down"].between(1, 4)
        & (cleaned["ydstogo"] > 0)
        & cleaned["yardline_100"].between(1, 99)
    ]

    cleaned["down"] = cleaned["down"].astype(int)

    cleaned["target"] = cleaned["play_type"].map({
        "run": 0,
        "pass": 1
    })

    if cleaned.empty:
        raise ValueError("No valid run or pass plays remain.")

    return cleaned.reset_index(drop=True)


def prepare_features(data):
    features = data[FEATURES]
    target = data["target"]

    return features, target


if __name__ == "__main__":
    try:
        raw_data = load_data()
        cleaned_data = clean_data(raw_data)

        features, target = prepare_features(cleaned_data)

        print(f"\nOriginal rows: {len(raw_data)}")
        print(f"Cleaned rows: {len(cleaned_data)}")
        print(f"Removed rows: {len(raw_data) - len(cleaned_data)}")

        print("\nPlay counts:")
        print(cleaned_data["play_type"].value_counts())

        print("\nPlay percentages:")
        print(
            cleaned_data["play_type"]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print("\nMissing values remaining:")
        print(cleaned_data.isna().sum())

        print("\nSample cleaned data:")
        print(cleaned_data.head(10).to_string(index=False))

        print(f"\nFeature shape: {features.shape}")
        print(f"Target shape: {target.shape}")

    except Exception as error:
        print(f"Preprocessing failed: {error}")
        raise SystemExit(1)