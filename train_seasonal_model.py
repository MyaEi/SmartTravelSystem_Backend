import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("seasonality_train.parquet")
MODEL_PATH = Path("seasonal_popularity.joblib")

# --------------------------------------------------
# 1️⃣ Load data
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError("❌ seasonality_train.parquet not found — run the backend first.")

df = pd.read_parquet(DATA_PATH)
print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
print(df.head())

# --------------------------------------------------
# 2️⃣ Ensure all expected columns exist
# --------------------------------------------------
expected_cols = ["destination", "name", "type", "category", "month", "source", "season", "country"]
for col in expected_cols:
    if col not in df.columns:
        df[col] = None

# drop rows missing key info
df = df.dropna(subset=["category", "season", "country"], how="any")

# --------------------------------------------------
# 3️⃣ Feature engineering
# --------------------------------------------------

# Add binary “has_image” flag (simulate availability)
df["has_image"] = np.random.randint(0, 2, len(df))

# Simulate missing rating column if needed
if "rating" not in df.columns:
    df["rating"] = np.random.uniform(3.5, 5.0, len(df))  # pseudo-rating

# Compute a synthetic target: "popularity"
df["popularity"] = (df["rating"].fillna(0) * 20) + df["has_image"].astype(int) * 10
print("🧮 Created synthetic target column 'popularity'")

# --------------------------------------------------
# 4️⃣ Encode categorical features
# --------------------------------------------------
for col in ["destination", "type", "category", "season", "country"]:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# --------------------------------------------------
# 5️⃣ Train/test split
# --------------------------------------------------
X = df[["destination", "type", "category", "season", "country", "has_image", "rating"]]
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# 6️⃣ Train model
# --------------------------------------------------
print("🧠 Training RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)
print("✅ Model training completed!")

# --------------------------------------------------
# 7️⃣ Evaluate quickly
# --------------------------------------------------
score = model.score(X_test, y_test)
print(f"📊 R² score: {score:.3f}")

# --------------------------------------------------
# 8️⃣ Save model
# --------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved as → {MODEL_PATH.resolve()}")
