from pathlib import Path

import nflreadpy as nfl
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(season=2025):
    DATA_DIR.mkdir(exist_ok=True)
    file_path = DATA_DIR / f"play_by_play_{season}.parquet"

    # Cache the dataset locally to avoid repeated downloads.
    if file_path.exists():
        print(f"Loading cached data for {season}...")
        return pd.read_parquet(file_path)

    print(f"Downloading NFL play-by-play data for {season}...")

    data = nfl.load_pbp(season).to_pandas()

    data.to_parquet(file_path, index=False)
    print(f"Data saved to {file_path}")

    return data


if __name__ == "__main__":
    try:
        data = load_data()

        print(f"\nNumber of rows: {len(data)}")
        print(f"Number of columns: {len(data.columns)}")

        columns = [
            "game_id",
            "posteam",
            "down",
            "ydstogo",
            "yardline_100",
            "play_type"
        ]

        print("\nSample data:")
        print(data[columns].head(10).to_string(index=False))

    except Exception as error:
        print(f"Unable to load NFL data: {error}")
        raise SystemExit(1)