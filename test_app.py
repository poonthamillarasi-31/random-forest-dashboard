"""
Automated Validation and Test Suite for Fashion Trend Prediction Dashboard
"""

import unittest
import pandas as pd
import numpy as np
from PIL import Image

from src.data_engine import (
    generate_fashion_timeseries,
    generate_social_buzz_posts,
    generate_merchandise_sku_data,
    get_cached_or_generate_dataset
)
from src.forecast_engine import (
    fit_trend_forecast,
    decompose_seasonality,
    compare_multiple_trends
)
from src.nlp_social import (
    analyze_hashtag_velocity,
    get_sentiment_distribution,
    extract_keywords_and_frequencies
)
from src.vision_extractor import (
    extract_dominant_colors_from_image,
    analyze_color_harmony,
    create_sample_fashion_image
)
from src.recommendation import (
    calculate_merchandise_strategy,
    generate_budget_reallocation_summary
)
from src.ai_copilot import (
    ask_fashion_copilot,
    run_trend_scenario_simulation
)


class TestFashionTrendPipeline(unittest.TestCase):

    def setUp(self):
        self.ts_df = generate_fashion_timeseries(start_date="2024-01-01", end_date="2025-06-01")
        self.social_df = generate_social_buzz_posts()
        self.sku_df = generate_merchandise_sku_data()

    def test_data_generation(self):
        self.assertFalse(self.ts_df.empty)
        self.assertIn("Search_Index", self.ts_df.columns)
        self.assertIn("Aesthetic", self.ts_df.columns)
        self.assertFalse(self.social_df.empty)
        self.assertFalse(self.sku_df.empty)

    def test_forecasting_engine(self):
        sub_df = self.ts_df[self.ts_df["Aesthetic"] == "Quiet Luxury / Minimalist"]
        fc_df, metrics = fit_trend_forecast(sub_df, forecast_horizon_months=6)
        self.assertEqual(metrics["status"], "Success")
        self.assertIn("Forecast", fc_df.columns)
        self.assertTrue((fc_df["Is_Forecast"] == True).sum() > 0)

        # Test Seasonality
        season_df = decompose_seasonality(sub_df)
        self.assertFalse(season_df.empty)

        # Test Comparison
        comp_df = compare_multiple_trends(self.ts_df, group_col="Aesthetic", forecast_horizon_months=3)
        self.assertFalse(comp_df.empty)

    def test_nlp_social(self):
        hashtags = analyze_hashtag_velocity(self.social_df)
        self.assertIn("Velocity_Index", hashtags.columns)
        
        sent = get_sentiment_distribution(self.social_df)
        self.assertIn("positive_pct", sent)
        
        words = extract_keywords_and_frequencies(self.social_df)
        self.assertIsInstance(words, dict)

    def test_vision_extractor(self):
        sample_img = create_sample_fashion_image("Quiet Luxury")
        self.assertIsInstance(sample_img, Image.Image)

        extracted = extract_dominant_colors_from_image(sample_img, num_colors=4)
        self.assertEqual(len(extracted), 4)
        self.assertTrue(extracted[0]["hex"].startswith("#"))

        harmony = analyze_color_harmony(extracted)
        self.assertIn("harmony_type", harmony)

    def test_recommendation_engine(self):
        strat = calculate_merchandise_strategy(self.sku_df)
        self.assertIn("Priority_Score", strat.columns)
        self.assertIn("Recommended_Action", strat.columns)

        budget = generate_budget_reallocation_summary(self.sku_df)
        self.assertIn("top_growth_aesthetic", budget)

    def test_ai_copilot(self):
        res = ask_fashion_copilot("What are the top aesthetics right now?", self.ts_df, self.sku_df)
        self.assertEqual(res["status"], "Success")
        self.assertIn("Top 3", res["response"])

        sim = run_trend_scenario_simulation("Quiet Luxury / Minimalist", 20.0, self.ts_df)
        self.assertEqual(sim["shock_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
