import pandas as pd

# --------------------------------------------------
# BASIC CAPACITY-BASED RISK SEVERITY
# --------------------------------------------------

def assign_risk_severity(capacity_gap):
    if capacity_gap <= 0:
        return "LOW"
    elif capacity_gap <= 10:
        return "MEDIUM"
    elif capacity_gap <= 25:
        return "HIGH"
    else:
        return "CRITICAL"


def detect_capacity_risk(df, buffer_ratio=1.1):
    """
    Detect risk severity based on demand vs capacity
    """
    df = df.copy()

    df["capacity_gap"] = (
        df["demand"] - df["estimated_capacity"] * buffer_ratio
    )

    df["risk_severity"] = df["capacity_gap"].apply(assign_risk_severity)

    return df


# --------------------------------------------------
# UNCERTAINTY-AWARE RISK ESCALATION (WORST CASE)
# --------------------------------------------------

def detect_risk_with_uncertainty(df, buffer_ratio=1.1):
    """
    Escalate risk using forecast upper bound
    """
    df = df.copy()

    # Safety check (important for Streamlit)
    required_cols = ["forecast_upper", "estimated_capacity"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["worst_case_gap"] = (
        df["forecast_upper"] - df["estimated_capacity"] * buffer_ratio
    )

    def uncertainty_severity(x):
        if x <= 0:
            return "LOW"
        elif x <= 10:
            return "MEDIUM"
        elif x <= 25:
            return "HIGH"
        else:
            return "CRITICAL"

    df["uncertainty_aware_risk"] = df["worst_case_gap"].apply(uncertainty_severity)

    return df


# --------------------------------------------------
# ALERT FATIGUE CONTROL (COOLDOWN LOGIC)
# --------------------------------------------------

def suppress_redundant_alerts(df, cooldown_days=3):
    """
    Prevent repeated alerts when severity does not change
    """
    df = df.copy()

    # Ensure datetime
    df["date"] = pd.to_datetime(df["date"])

    df["alert_allowed"] = True

    last_alert_date = None
    last_severity = None

    for i, row in df.iterrows():
        severity = row["risk_severity"]
        current_date = row["date"]  # datetime guaranteed

        if severity in ["HIGH", "CRITICAL"]:
            if last_alert_date is not None:
                days_diff = (current_date - last_alert_date).days

                if days_diff <= cooldown_days and severity == last_severity:
                    df.at[i, "alert_allowed"] = False
                else:
                    last_alert_date = current_date
                    last_severity = severity
            else:
                last_alert_date = current_date
                last_severity = severity

    return df



# --------------------------------------------------
# ROOT CAUSE ATTRIBUTION
# --------------------------------------------------

def assign_root_cause(df):
    """
    Identify primary driver behind risk increase
    """
    df = df.copy()

    avg_resources = df["active_resources"].mean()
    high_backlog = df["backlog"].quantile(0.75)

    def root_cause(row):
        if row["demand"] > row["rolling_mean_7"] * 1.15:
            return "Demand Spike"
        elif row["active_resources"] < avg_resources:
            return "Resource Drop"
        elif row["backlog"] > high_backlog:
            return "Backlog Accumulation"
        else:
            return "Mixed Factors"

    df["root_cause"] = df.apply(root_cause, axis=1)

    return df


# --------------------------------------------------
# LOCAL TEST (OPTIONAL)
# --------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("data/processed/forecast_features.csv")
    df["estimated_capacity"] = 120  # dummy for local test

    # Dummy forecast upper for test
    df["forecast_upper"] = df["demand"] + 15

    df = detect_capacity_risk(df)
    df = detect_risk_with_uncertainty(df)
    df = suppress_redundant_alerts(df)
    df = assign_root_cause(df)

    print(
        df[[
            "date",
            "demand",
            "estimated_capacity",
            "risk_severity",
            "uncertainty_aware_risk",
            "alert_allowed",
            "root_cause"
        ]].tail()
    )
