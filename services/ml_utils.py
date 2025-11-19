import pandas as pd
from pathlib import Path

DATA_LOG_PATH = Path("seasonality_train.parquet")

def log_training_data(destination: str, results: dict, season: str = "", country: str = ""):
    """
    Append attraction and event data to the local training dataset.
    """
    print(f"🟡 [DEBUG] Logging training data for destination: {destination}")

    attractions = results.get("attractions", [])
    events = results.get("events", [])
    records = []

    for item in attractions + events:
        if isinstance(item, dict):
            records.append({
                "destination": destination,
                "name": item.get("name", "unknown"),
                "type": item.get("type", "unknown"),
                "category": item.get("category", "general"),
                "month": item.get("date", None),
                "source": item.get("source", None),
                "season": season,
                "country": country,
            })
        else:
            records.append({
                "destination": destination,
                "name": str(item),
                "type": "unknown",
                "category": "general",
                "month": None,
                "source": None,
                "season": season,
                "country": country,
            })

    print(f"🟢 [DEBUG] Collected {len(records)} records")

    if not records:
        print(f"⚠️ [DEBUG] No records found for {destination}")
        return

    new_df = pd.DataFrame(records)

    try:
        if DATA_LOG_PATH.exists():
            old_df = pd.read_parquet(DATA_LOG_PATH)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_parquet(DATA_LOG_PATH, index=False)
        print(f"✅ [DEBUG] Saved {len(records)} rows → {DATA_LOG_PATH.resolve()}")

    except Exception as e:
        print(f"❌ [ERROR] Could not write parquet: {e}")
