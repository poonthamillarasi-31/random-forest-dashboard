"""
Recommendation Engine for Retail Merchandising & Inventory Planning
Translates trend trajectories, sell-through velocity, and volatility
into actionable purchasing, restocking, and markdown recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def calculate_merchandise_strategy(sku_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies multi-factor scoring on SKUs using Forecast Growth, Margin, and Stock levels
    to assign recommended order adjustments and priority rank.
    """
    df = sku_df.copy()

    # Priority score formula: (Forecast Growth * 0.5) + (Gross Margin * 0.3) + (Sell-Through * 0.2)
    norm_growth = np.clip(df["Forecast_Growth_Pct"], -50, 100)
    df["Priority_Score"] = (
        (norm_growth * 0.5) +
        (df["Gross_Margin_Pct"] * 0.3) +
        (df["Sell_Through_Rate"] * 0.2)
    ).round(1)

    # Order adjustment multiplier
    def get_adjustment_units(row):
        growth = row["Forecast_Growth_Pct"]
        current_stock = row["Current_Stock_Units"]
        if growth >= 25:
            return int(current_stock * 0.40)  # Restock +40%
        elif growth >= 10:
            return int(current_stock * 0.20)  # Restock +20%
        elif growth >= -5:
            return 0  # Reorder as needed
        else:
            return -int(current_stock * 0.30)  # Reduce / Markdown

    df["Suggested_Unit_Delta"] = df.apply(get_adjustment_units, axis=1)
    df["Recommended_Action"] = df["Suggested_Unit_Delta"].apply(
        lambda x: f"Increase Buy by +{x} units" if x > 0 else (f"Markdown / Reduce by {abs(x)} units" if x < 0 else "Maintain Steady Replenishment")
    )

    return df.sort_values("Priority_Score", ascending=False)


def generate_budget_reallocation_summary(sku_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes aggregate strategic budget shifts across fashion aesthetics and categories.
    """
    aesthetic_summary = sku_df.groupby("Aesthetic").agg(
        Total_Stock_Units=("Current_Stock_Units", "sum"),
        Avg_Forecast_Growth=("Forecast_Growth_Pct", "mean"),
        Avg_Margin=("Gross_Margin_Pct", "mean")
    ).reset_index()

    aesthetic_summary["Recommended_Budget_Shift"] = aesthetic_summary["Avg_Forecast_Growth"].apply(
        lambda g: f"+{int(g * 0.8)}% Budget Expansion" if g > 5 else (f"{int(g * 0.8)}% Budget Contraction" if g < -5 else "Neutral Budget Allocation")
    )
    aesthetic_summary["Avg_Forecast_Growth"] = aesthetic_summary["Avg_Forecast_Growth"].round(1)
    aesthetic_summary["Avg_Margin"] = aesthetic_summary["Avg_Margin"].round(1)

    top_growing = aesthetic_summary.sort_values("Avg_Forecast_Growth", ascending=False).iloc[0]["Aesthetic"]
    most_at_risk = aesthetic_summary.sort_values("Avg_Forecast_Growth", ascending=True).iloc[0]["Aesthetic"]

    return {
        "summary_table": aesthetic_summary,
        "top_growth_aesthetic": top_growing,
        "most_at_risk_aesthetic": most_at_risk
    }
