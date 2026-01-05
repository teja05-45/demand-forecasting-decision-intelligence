import pandas as pd
import numpy as np

def generate_demand_data(
    start_date="2022-01-01",
    days=730
):
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=days, freq="D")

    base_demand = 120
    seasonal = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
    trend = np.linspace(0, 30, days)
    noise = np.random.normal(0, 10, days)

    demand = base_demand + seasonal + trend + noise
    demand = np.maximum(demand, 20).astype(int)

    active_resources = np.random.randint(15, 25, size=days)
    avg_resolution_time = np.random.normal(4, 0.5, days)
    backlog = np.maximum(demand - active_resources * 6, 0)

    df = pd.DataFrame({
        "date": dates,
        "demand": demand,
        "avg_resolution_time": avg_resolution_time,
        "active_resources": active_resources,
        "backlog": backlog
    })

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["demand_growth_rate"] = df["demand"].pct_change().fillna(0)

    return df


if __name__ == "__main__":
    df = generate_demand_data()
    df.to_csv("data/raw/demand_data.csv", index=False)
    print("✅ Demand dataset generated successfully")
