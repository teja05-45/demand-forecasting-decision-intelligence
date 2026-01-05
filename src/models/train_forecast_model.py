import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os


def train_forecast_model():
    # Load processed features
    df = pd.read_csv("data/processed/forecast_features.csv")

    # Define features & target
    target = "demand"
    features = [
        "demand_lag_1",
        "demand_lag_7",
        "demand_lag_14",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "day_of_week",
        "is_weekend",
        "avg_resolution_time",
        "active_resources",
        "backlog",
        "demand_growth_rate"
    ]

    X = df[features]
    y = df[target]

    # Time-based split (last 20% for test)
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"✅ Forecast Model MAE: {mae:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/demand_forecast_model.pkl")
    print("✅ Model saved successfully")


if __name__ == "__main__":
    train_forecast_model()
