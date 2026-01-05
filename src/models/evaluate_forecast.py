import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error


def forecast_with_uncertainty(df, confidence=0.9):
    model = joblib.load("models/demand_forecast_model.pkl")

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

    df = df.copy()
    df["forecast"] = model.predict(df[features])

    # Estimate residual error
    residual_std = np.std(df["demand"] - df["forecast"])

    z = 1.65 if confidence == 0.9 else 1.96

    df["forecast_lower"] = df["forecast"] - z * residual_std
    df["forecast_upper"] = df["forecast"] + z * residual_std

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/forecast_features.csv")
    df = forecast_with_uncertainty(df)
    print(df[["forecast", "forecast_lower", "forecast_upper"]].tail())
