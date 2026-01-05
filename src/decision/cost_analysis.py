import numpy as np
def calculate_expected_cost(
    df,
    sla_penalty_cost=500,
    idle_resource_cost=100
):
    df = df.copy()

    df["sla_risk_cost"] = (
        df["risk_severity"]
        .map({
            "LOW": 0,
            "MEDIUM": 0.3,
            "HIGH": 0.7,
            "CRITICAL": 1.0
        }) * sla_penalty_cost
    )

    df["idle_capacity"] = (
        df["estimated_capacity"] - df["demand"]
    ).clip(lower=0)

    df["idle_cost"] = df["idle_capacity"] * idle_resource_cost

    df["total_expected_cost"] = (
        df["sla_risk_cost"] + df["idle_cost"]
    )

    return df



def find_optimal_buffer(
    df,
    buffer_candidates=[1.0, 1.05, 1.1, 1.15, 1.2],
    sla_penalty_cost=500,
    idle_resource_cost=100
):
    results = []

    for buffer in buffer_candidates:
        temp = df.copy()
        temp["capacity_gap"] = (
            temp["forecast_upper"] - temp["estimated_capacity"] * buffer
        )

        sla_cost = (
            (temp["capacity_gap"] > 0).astype(int) * sla_penalty_cost
        ).sum()

        idle_cost = (
            (temp["estimated_capacity"] - temp["forecast"]).clip(lower=0)
            * idle_resource_cost
        ).sum()

        total_cost = sla_cost + idle_cost

        results.append((buffer, total_cost))

    return min(results, key=lambda x: x[1])

