import pandas as pd
from src.decision.capacity_model import estimate_capacity
from src.decision.risk_detection import detect_capacity_risk



def run_what_if_scenario(
    df,
    demand_change_pct=0.0,
    resource_change=0,
    tickets_per_resource_per_day=6
):
    """
    Simulate what-if scenarios by adjusting demand and capacity
    """

    df = df.copy()

    # Apply demand change
    df["simulated_demand"] = (
        df["demand"] * (1 + demand_change_pct)
    ).round()

    # Apply resource change
    df["simulated_resources"] = (
        df["active_resources"] + resource_change
    ).clip(lower=1)

    # Recalculate capacity
    df["simulated_capacity"] = (
        df["simulated_resources"] * tickets_per_resource_per_day
    )

    # Risk detection
    df["simulated_risk_flag"] = (
        df["simulated_demand"] > df["simulated_capacity"] * 1.1
    ).astype(int)

    return df
def compare_scenarios(df, scenarios):
    results = []

    for name, params in scenarios.items():
        sim_df = run_what_if_scenario(
            df,
            demand_change_pct=params["demand_change"],
            resource_change=params["resource_change"]
        )

        critical_days = sim_df[
            sim_df["simulated_risk_flag"] == 1
        ].shape[0]

        results.append({
            "Scenario": name,
            "Critical_Days": critical_days
        })

    return pd.DataFrame(results)

def backtest_resource_decision(
    df,
    resource_change=3
):
    simulated = run_what_if_scenario(
        df,
        demand_change_pct=0.0,
        resource_change=resource_change
    )

    actual_risk_days = (
        df["risk_severity"].isin(["HIGH", "CRITICAL"]).sum()
    )

    simulated_risk_days = (
        simulated["simulated_risk_flag"].sum()
    )

    return {
        "actual_risk_days": actual_risk_days,
        "simulated_risk_days": simulated_risk_days,
        "risk_days_avoided": actual_risk_days - simulated_risk_days
    }


if __name__ == "__main__":
    df = pd.read_csv("data/processed/forecast_features.csv")

    # Example: demand +20%, add 3 resources
    scenario_df = run_what_if_scenario(
        df,
        demand_change_pct=0.20,
        resource_change=3
    )

    print(
        scenario_df[
            ["date", "demand", "simulated_demand",
             "active_resources", "simulated_resources",
             "simulated_capacity", "simulated_risk_flag"]
        ].tail()
    )
