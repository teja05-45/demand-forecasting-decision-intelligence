import pandas as pd


def estimate_capacity(df, tickets_per_resource_per_day=6):
    """
    Estimate daily handling capacity based on available resources
    """
    df = df.copy()
    df["estimated_capacity"] = (
        df["active_resources"] * tickets_per_resource_per_day
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/forecast_features.csv")
    df = estimate_capacity(df)
    print(df[["date", "active_resources", "estimated_capacity"]].head())
