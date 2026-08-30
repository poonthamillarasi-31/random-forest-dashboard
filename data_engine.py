"""
Data Engine for Fashion Trend Prediction
Generates, loads, and manages realistic fashion trend time series,
social sentiment, runway attributes, color data, and inventory metrics.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fashion Domain Knowledge Constants
FASHION_CATEGORIES = [
    "Tailoring & Blazers",
    "Knitwear & Sweaters",
    "Outerwear & Trench Coats",
    "Dresses & Eveningwear",
    "Denim & Wide-Leg Pants",
    "Streetwear & Graphic Tees",
    "Footwear & Chunky Loafers",
    "Leather & Suede Jackets",
    "Athleisure & Loungewear",
    "Accessories & Statement Bags",
]

AESTHETICS = [
    "Quiet Luxury / Minimalist",
    "Y2K Nostalgia",
    "Gorpcore & Utility",
    "Cottagecore & Bohemian",
    "Eclectic Grandpa / Retro",
    "Mob Wife Glamour",
    "Cyberpunk & Techwear",
    "Clean Girl Aesthetic",
]

TRENDING_COLORS = [
    {"name": "Sage Green", "hex": "#8A9A86", "family": "Green", "season": "Spring/Summer"},
    {"name": "Digital Lavender", "hex": "#B5A7D6", "family": "Purple", "season": "All Season"},
    {"name": "Fiery Cherry Red", "hex": "#BA131A", "family": "Red", "season": "Fall/Winter"},
    {"name": "Mocha & Espresso Brown", "hex": "#4B3728", "family": "Brown", "season": "Fall/Winter"},
    {"name": "Cobalt Blue", "hex": "#0047AB", "family": "Blue", "season": "All Season"},
    {"name": "Metallic Silver & Chrome", "hex": "#C0C0C0", "family": "Metallic", "season": "Fall/Winter"},
    {"name": "Butter Yellow", "hex": "#FFF1A8", "family": "Yellow", "season": "Spring/Summer"},
    {"name": "Peach Fuzz", "hex": "#FFBE98", "family": "Orange", "season": "Spring/Summer"},
    {"name": "Optic White", "hex": "#F8F9FA", "family": "Neutral", "season": "All Season"},
    {"name": "Obsidian Black", "hex": "#1C1C1E", "family": "Neutral", "season": "All Season"},
]

FABRICS = [
    "Organic Heavyweight Linen",
    "Cashmere & Merino Blend",
    "Raw Selvedge Denim",
    "Cruelty-Free Vegan Leather",
    "Liquid Silk & Satin",
    "Recycled Technical Polyamide",
    "Chunky Mohair Knit",
    "Sheer Organza & Mesh",
]

CITIES = ["Paris", "Milan", "New York", "London", "Tokyo", "Seoul"]
DEMOGRAPHICS = ["Gen Z (16-26)", "Millennials (27-42)", "Gen X & Prime (43-58)", "Luxury Connoisseurs"]


def generate_fashion_timeseries(
    start_date="2023-01-01",
    end_date="2026-08-01",
    freq="MS",
    random_seed=42
) -> pd.DataFrame:
    """
    Generates monthly historical trend search volume, runway presence score,
    and retail consumer interest index across all categories and aesthetics.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    records = []

    for dt in date_range:
        month = dt.month
        year = dt.year
        t = (dt - pd.to_datetime(start_date)).days / 365.0  # Time progress in years

        # Seasonal Factors
        ss_seasonality = np.sin(2 * np.pi * (month - 3) / 12)  # Peaks in Spring/Summer (June)
        fw_seasonality = np.cos(2 * np.pi * (month - 1) / 12)  # Peaks in Fall/Winter (Dec/Jan)

        for category in FASHION_CATEGORIES:
            for aesthetic in AESTHETICS:
                # Specific trajectory patterns for different aesthetics
                if aesthetic == "Quiet Luxury / Minimalist":
                    base_trend = 50 + 22 * t + 8 * fw_seasonality
                elif aesthetic == "Mob Wife Glamour":
                    # Spiked in 2024, stabilizing
                    base_trend = 25 + 35 * np.exp(-((t - 1.5) ** 2) / 0.4) + 12 * fw_seasonality
                elif aesthetic == "Eclectic Grandpa / Retro":
                    base_trend = 30 + 18 * t + 6 * np.sin(t * 3)
                elif aesthetic == "Gorpcore & Utility":
                    base_trend = 45 + 12 * t + 15 * fw_seasonality
                elif aesthetic == "Y2K Nostalgia":
                    base_trend = 65 - 8 * t + 10 * ss_seasonality
                elif aesthetic == "Clean Girl Aesthetic":
                    base_trend = 55 + 5 * t + 8 * ss_seasonality
                else:
                    base_trend = 40 + 10 * np.sin(t * 2) + 5 * ss_seasonality

                # Category adjustments
                if "Outerwear" in category or "Leather" in category:
                    category_factor = 1.2 if month in [10, 11, 12, 1, 2] else 0.6
                elif "Dresses" in category or "Denim" in category:
                    category_factor = 1.3 if month in [4, 5, 6, 7, 8] else 0.7
                elif "Knitwear" in category:
                    category_factor = 1.4 if month in [10, 11, 12, 1] else 0.5
                else:
                    category_factor = 1.0

                noise = np.random.normal(0, 4)
                search_index = np.clip((base_trend * category_factor) + noise, 5, 100)
                sales_volume = int(search_index * np.random.uniform(120, 240))
                runway_share_pct = np.clip((search_index / 10) + np.random.normal(0, 1.2), 0.5, 25.0)
                social_mentions_k = round((search_index * np.random.uniform(3.5, 9.2)), 1)
                sentiment_score = np.clip(0.55 + 0.35 * np.sin(search_index / 20) + np.random.normal(0, 0.08), 0.1, 0.98)

                records.append({
                    "Date": dt,
                    "Year": year,
                    "Month": month,
                    "Category": category,
                    "Aesthetic": aesthetic,
                    "Search_Index": round(float(search_index), 2),
                    "Sales_Volume": sales_volume,
                    "Runway_Share_Pct": round(float(runway_share_pct), 2),
                    "Social_Mentions_K": social_mentions_k,
                    "Sentiment_Score": round(float(sentiment_score), 3),
                    "Primary_Region": np.random.choice(CITIES),
                    "Core_Demographic": np.random.choice(DEMOGRAPHICS)
                })

    df = pd.DataFrame(records)
    return df


def generate_social_buzz_posts() -> pd.DataFrame:
    """
    Generates simulated real-time social media posts with hashtags,
    engagement metrics, influencer tier, and sentiment for NLP analysis.
    """
    hashtags_pool = [
        ("#QuietLuxury", "Quiet Luxury / Minimalist", 0.88),
        ("#OldMoneyStyle", "Quiet Luxury / Minimalist", 0.84),
        ("#EclecticGrandpa", "Eclectic Grandpa / Retro", 0.91),
        ("#MobWifeAesthetic", "Mob Wife Glamour", 0.76),
        ("#GorpcoreOutfit", "Gorpcore & Utility", 0.82),
        ("#Y2KFashion", "Y2K Nostalgia", 0.72),
        ("#CleanGirlAesthetic", "Clean Girl Aesthetic", 0.79),
        ("#TechwearStyle", "Cyberpunk & Techwear", 0.85),
        ("#CherryRedTrend", "Fiery Cherry Red", 0.93),
        ("#ButterYellowSummer", "Butter Yellow", 0.89),
        ("#OOTDRunway", "Runway General", 0.81),
        ("#ThriftFlipRetro", "Eclectic Grandpa / Retro", 0.87),
        ("#LinenSeason", "Quiet Luxury / Minimalist", 0.90),
        ("#LeatherWeather", "Outerwear & Trench Coats", 0.86),
    ]

    sample_captions = [
        "Obsessed with this minimalist tailoring for FW26! The silhouette is perfection ✨ {tag} {tag2}",
        "Vintage knitwear is truly taking over my wardrobe this season. Loving the rich textures {tag}",
        "Runway recap: bold cherry red accents paired with deep espresso tones are the ultimate combo! {tag} {tag2}",
        "Utility outerwear styled with sleek tailored trousers. Best streetwear blend right now {tag}",
        "Effortless elegance with silk slip dresses and oversized structured blazers {tag} {tag2}",
        "Is 90s minimalism coming back even stronger? Neutral tones and sharp tailoring everywhere {tag}",
        "Statement chunky footwear paired with wide-leg selvedge denim. Instant outfit upgrade {tag}",
        "The shift towards tactile fabrics like mohair and raw linen is unmistakable {tag} {tag2}",
        "Loving the subtle metallic silver details on outerwear and accessories {tag}",
        "Capsule wardrobe essential: timeless cashmere sweaters and tailored trench coats {tag}"
    ]

    np.random.seed(101)
    posts = []
    platforms = ["TikTok", "Instagram", "Pinterest", "X / Twitter", "Threads"]
    influencer_tiers = ["Nano (1k-10k)", "Micro (10k-50k)", "Macro (50k-500k)", "Mega / Celebrity (500k+)"]

    for i in range(250):
        tag_info1 = hashtags_pool[np.random.randint(0, len(hashtags_pool))]
        tag_info2 = hashtags_pool[np.random.randint(0, len(hashtags_pool))]
        template = np.random.choice(sample_captions)
        caption = template.format(tag=tag_info1[0], tag2=tag_info2[0] if tag_info2[0] != tag_info1[0] else "#FashionForecast")
        
        likes = int(np.random.exponential(scale=4500) + 150)
        shares = int(likes * np.random.uniform(0.05, 0.25))
        comments = int(likes * np.random.uniform(0.02, 0.12))
        sentiment_val = np.clip(np.random.normal((tag_info1[2] + tag_info2[2]) / 2, 0.12), 0.1, 0.99)
        
        if sentiment_val >= 0.70:
            sentiment_label = "Positive"
        elif sentiment_val >= 0.45:
            sentiment_label = "Neutral"
        else:
            sentiment_label = "Negative"

        post_date = datetime.now() - timedelta(days=np.random.randint(0, 90), hours=np.random.randint(0, 24))

        posts.append({
            "Post_ID": f"SOC-{1000 + i}",
            "Platform": np.random.choice(platforms),
            "Date": post_date.strftime("%Y-%m-%d"),
            "Caption": caption,
            "Primary_Hashtag": tag_info1[0],
            "Aesthetic_Mapped": tag_info1[1],
            "Likes": likes,
            "Shares": shares,
            "Comments": comments,
            "Engagement_Score": likes + (shares * 3) + (comments * 2),
            "Sentiment_Score": round(float(sentiment_val), 3),
            "Sentiment_Label": sentiment_label,
            "Influencer_Tier": np.random.choice(influencer_tiers, p=[0.4, 0.35, 0.2, 0.05]),
            "City": np.random.choice(CITIES)
        })

    return pd.DataFrame(posts)


def generate_merchandise_sku_data() -> pd.DataFrame:
    """
    Generates SKU-level retail inventory data with production cost,
    selling price, stock levels, forecast trend momentum, and risk tier.
    """
    np.random.seed(88)
    skus = []

    sku_templates = [
        ("Oversized Double-Breasted Blazer", "Tailoring & Blazers", "Quiet Luxury / Minimalist", "Sage Green", "Organic Heavyweight Linen", 180, 420),
        ("Chunky Cable-Knit Mohair Sweater", "Knitwear & Sweaters", "Eclectic Grandpa / Retro", "Butter Yellow", "Chunky Mohair Knit", 120, 290),
        ("Cropped Moto Vegan Leather Jacket", "Leather & Suede Jackets", "Mob Wife Glamour", "Obsidian Black", "Cruelty-Free Vegan Leather", 140, 350),
        ("Relaxed Wide-Leg Raw Denim", "Denim & Wide-Leg Pants", "Clean Girl Aesthetic", "Cobalt Blue", "Raw Selvedge Denim", 75, 195),
        ("Floor-Length Silk Slip Dress", "Dresses & Eveningwear", "Quiet Luxury / Minimalist", "Peach Fuzz", "Liquid Silk & Satin", 110, 310),
        ("Weatherproof Utility Shell Trench", "Outerwear & Trench Coats", "Gorpcore & Utility", "Sage Green", "Recycled Technical Polyamide", 195, 480),
        ("Lug-Sole Platform Leather Loafer", "Footwear & Chunky Loafers", "Eclectic Grandpa / Retro", "Mocha & Espresso Brown", "Cruelty-Free Vegan Leather", 95, 260),
        ("Metallic Chrome Puffer Coat", "Outerwear & Trench Coats", "Cyberpunk & Techwear", "Metallic Silver & Chrome", "Recycled Technical Polyamide", 160, 390),
        ("Cashmere Turtleneck Sweater", "Knitwear & Sweaters", "Quiet Luxury / Minimalist", "Mocha & Espresso Brown", "Cashmere & Merino Blend", 150, 380),
        ("Graphic Distressed Baby Tee", "Streetwear & Graphic Tees", "Y2K Nostalgia", "Fiery Cherry Red", "Organic Heavyweight Linen", 25, 75),
    ]

    for idx, (name, cat, aes, color, fabric, cost, price) in enumerate(sku_templates):
        current_stock = np.random.randint(120, 850)
        lead_time_weeks = np.random.choice([4, 6, 8, 12])
        sell_through_rate = np.random.uniform(0.55, 0.94)
        trend_momentum = np.random.uniform(-15.0, 45.0)  # % growth predicted
        
        # Risk assessment
        if trend_momentum > 20:
            recommendation = "Aggressive Restock / Scale Up (+30% to +50%)"
            risk_tier = "Low (High Demand Surge)"
        elif trend_momentum > 5:
            recommendation = "Moderate Reorder (+10% to +20%)"
            risk_tier = "Low-Medium (Stable Growth)"
        elif trend_momentum > -5:
            recommendation = "Maintain Current Stock Velocity"
            risk_tier = "Medium (Plateauing Trend)"
        else:
            recommendation = "Phased Markdown / Liquidate Remaining Stock"
            risk_tier = "High (Declining Momentum)"

        skus.append({
            "SKU_Code": f"SKU-{cat[:3].upper()}-{202600 + idx}",
            "Item_Name": name,
            "Category": cat,
            "Aesthetic": aes,
            "Color_Palette": color,
            "Primary_Fabric": fabric,
            "Unit_Cost_USD": cost,
            "Retail_Price_USD": price,
            "Gross_Margin_Pct": round(((price - cost) / price) * 100, 1),
            "Current_Stock_Units": current_stock,
            "Sell_Through_Rate": round(sell_through_rate * 100, 1),
            "Forecast_Growth_Pct": round(trend_momentum, 1),
            "Lead_Time_Weeks": lead_time_weeks,
            "Procurement_Recommendation": recommendation,
            "Risk_Tier": risk_tier
        })

    return pd.DataFrame(skus)


def get_cached_or_generate_dataset(data_dir: str = "sample_data") -> pd.DataFrame:
    """
    Retrieves the dataset from disk if available, or generates and saves it to disk.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "fashion_trends_seed.csv")

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            return df
        except Exception:
            pass

    # Generate and save
    df = generate_fashion_timeseries()
    df.to_csv(file_path, index=False)
    return df
