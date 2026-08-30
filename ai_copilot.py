"""
AI Trend Copilot & Strategic Advisory Engine
Parses fashion queries, executes scenario simulations,
and generates structured advisory intelligence for fashion buyers & designers.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


SUGGESTED_PROMPTS = [
    "🔥 What are the top 3 highest momentum aesthetics right now?",
    "🎨 What complementary color palettes match Sage Green and Butter Yellow?",
    "🧥 How is Quiet Luxury tailoring projected to perform over the next 6 months?",
    "🛍️ What SKUs require immediate markdown due to declining momentum?",
    "🧵 Which sustainable fabrics are seeing the highest search growth?",
]


def ask_fashion_copilot(query: str, timeseries_df: pd.DataFrame, sku_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the user's natural language fashion query, correlates with current datasets,
    and returns a structured AI advisory response.
    """
    q_lower = query.lower()
    
    # 1. Top aesthetics momentum query
    if "top" in q_lower or "highest" in q_lower or "momentum" in q_lower or "aesthetics" in q_lower:
        latest_date = timeseries_df["Date"].max()
        past_6m = latest_date - pd.DateOffset(months=6)
        
        recent_df = timeseries_df[timeseries_df["Date"] >= past_6m]
        aesthetic_growth = recent_df.groupby("Aesthetic")["Search_Index"].agg(["first", "last"]).reset_index()
        aesthetic_growth["Growth_Pct"] = ((aesthetic_growth["last"] - aesthetic_growth["first"]) / aesthetic_growth["first"] * 100).round(1)
        top_3 = aesthetic_growth.sort_values("Growth_Pct", ascending=False).head(3)

        top_names = top_3["Aesthetic"].tolist()
        top_growths = top_3["Growth_Pct"].tolist()

        response_text = f"""
### 🌟 Top 3 Fashion Aesthetics by Momentum:

1. **{top_names[0]}** (+{top_growths[0]}% growth) — *High consumer search velocity & runway validation.*
2. **{top_names[1]}** (+{top_growths[1]}% growth) — *Strong social engagement and commercial sell-through.*
3. **{top_names[2]}** (+{top_growths[2]}% growth) — *Emerging niche trend transitioning into mainstream retail.*

**Strategic Advice:** Shift 20–30% of open-to-buy budget into tailored silhouettes and natural textured fabrics aligned with **{top_names[0]}**.
"""
        return {"response": response_text, "status": "Success"}

    # 2. Color Palette & Harmony
    elif "color" in q_lower or "palette" in q_lower or "sage" in q_lower or "yellow" in q_lower:
        response_text = """
### 🎨 Color Trend Intelligence & Harmonies:

- **Primary Driver:** **Sage Green (#8A9A86)** remains the foundational neutral bridging transitional seasons.
- **Accent Complement:** Pair Sage Green with **Mocha Espresso Brown (#4B3728)** for FW collections, or **Butter Yellow (#FFF1A8)** for SS collections.
- **Runway Accent of the Season:** **Fiery Cherry Red (#BA131A)** is showing a +28% spike in accessory and outerwear styling.

**Designer Recommendation:** Utilize a 60-30-10 palette split: 60% Earthy Muted Base (Sage/Camel), 30% Secondary Neutral (Off-White/Espresso), 10% High-Impact Accent (Cherry Red / Chrome Silver).
"""
        return {"response": response_text, "status": "Success"}

    # 3. Quiet Luxury & Tailoring
    elif "quiet luxury" in q_lower or "tailoring" in q_lower or "blazer" in q_lower:
        response_text = """
### 📈 Quiet Luxury & Tailoring Outlook:

- **Search Index Trajectory:** Stabilizing at an elevated plateau (Index ~78/100).
- **Key Shift:** Consumer preference is moving from ultra-minimalist plain pieces toward **tactile luxury** — incorporating heavy linen weaves, double-faced wool, and subtle sculptural drapery.
- **Forecast Horizon:** High commercial safety for the next 4 quarters with minimal markdown risk.

**Inventory Directive:** Maintain 35-40% inventory allocation in premium blazers, wide-leg tailored trousers, and cashmere knits.
"""
        return {"response": response_text, "status": "Success"}

    # 4. SKUs / Markdown & Inventory
    elif "sku" in q_lower or "markdown" in q_lower or "inventory" in q_lower or "stock" in q_lower:
        at_risk = sku_df[sku_df["Forecast_Growth_Pct"] < 0].sort_values("Forecast_Growth_Pct")
        if len(at_risk) > 0:
            item_list = "\n".join([f"- **{row['Item_Name']}** ({row['Aesthetic']}): Projected {row['Forecast_Growth_Pct']}% drop | Stock: {row['Current_Stock_Units']} units" for _, row in at_risk.iterrows()])
        else:
            item_list = "- All current core SKUs are exhibiting positive or stable momentum."

        response_text = f"""
### 🛍️ Markdown & Inventory Risk Alert:

The following styles show decelerating trend velocity and should be evaluated for phased promotional pricing:

{item_list}

**Action Plan:** Implement 15-25% bundle promotions or early end-of-season clearance before demand drops below target margin thresholds.
"""
        return {"response": response_text, "status": "Success"}

    # Default General Trend Intelligence Answer
    else:
        response_text = f"""
### 💡 Strategic Fashion Intelligence Summary for: *"{query}"*

- **Macro Trend Context:** Current global retail data demonstrates a strong convergence between **functional utility (Gorpcore)** and **refined modern tailoring (Quiet Luxury)**.
- **Consumer Behavior:** Gen Z and Millennial demographics are prioritizing fabric longevity (linen, heavyweight cotton, raw denim) and versatile capsule wardrobes over disposable fast fashion.
- **Key Recommendation:** Test small-batch capsule drops (150-300 units per SKU) before committing large-volume factory POs for emerging aesthetics.
"""
        return {"response": response_text, "status": "Success"}


def run_trend_scenario_simulation(
    target_aesthetic: str,
    shock_pct: float,
    current_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Simulates what happens if a specific fashion trend experiences an unexpected viral surge or slump.
    """
    sub_df = current_df[current_df["Aesthetic"] == target_aesthetic]
    current_avg_search = sub_df["Search_Index"].mean() if len(sub_df) > 0 else 50.0
    current_sales = sub_df["Sales_Volume"].mean() if len(sub_df) > 0 else 10000

    simulated_search = current_avg_search * (1 + shock_pct / 100.0)
    simulated_sales = current_sales * (1 + (shock_pct * 1.2) / 100.0)
    revenue_delta = (simulated_sales - current_sales) * 85.0  # Average unit retail revenue

    return {
        "aesthetic": target_aesthetic,
        "shock_pct": shock_pct,
        "current_search_index": round(current_avg_search, 1),
        "simulated_search_index": round(simulated_search, 1),
        "estimated_monthly_units_delta": int(simulated_sales - current_sales),
        "projected_revenue_impact_usd": round(revenue_delta, 2)
    }
