import pandas as pd


def create_time_series_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Lag features
    df["demand_lag_1"] = df["demand"].shift(1)
    df["demand_lag_7"] = df["demand"].shift(7)
    df["demand_lag_14"] = df["demand"].shift(14)

    # Rolling features
    df["rolling_mean_7"] = df["demand"].rolling(window=7).mean()
    df["rolling_std_7"] = df["demand"].rolling(window=7).std()
    df["rolling_mean_14"] = df["demand"].rolling(window=14).mean()

    # Drop rows with NaN created by lagging
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/demand_data.csv")
    df_features = create_time_series_features(df)
    df_features.to_csv("data/processed/forecast_features.csv", index=False)
    print("✅ Feature engineering completed successfully")
