"""
NLP and Social Media Buzz Analyzer
Performs sentiment breakdown, hashtag momentum tracking,
keyword frequency analysis, and WordCloud visualization.
"""

import io
import re
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def analyze_hashtag_velocity(social_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes aggregated hashtag metrics: Total Volume, Average Engagement,
    Average Sentiment, and Viral Velocity Index.
    """
    grouped = social_df.groupby("Primary_Hashtag").agg(
        Post_Count=("Post_ID", "count"),
        Total_Likes=("Likes", "sum"),
        Total_Shares=("Shares", "sum"),
        Total_Comments=("Comments", "sum"),
        Avg_Sentiment=("Sentiment_Score", "mean"),
        Avg_Engagement=("Engagement_Score", "mean")
    ).reset_index()

    # Calculate Viral Velocity Index: normalized engagement * post count * sentiment
    max_eng = grouped["Avg_Engagement"].max() if len(grouped) > 0 else 1
    grouped["Velocity_Index"] = (
        (grouped["Avg_Engagement"] / max_eng) * 50 + 
        (grouped["Post_Count"] / grouped["Post_Count"].max()) * 30 + 
        (grouped["Avg_Sentiment"] * 20)
    ).round(1)

    grouped["Avg_Sentiment_Score"] = grouped["Avg_Sentiment"].round(3)
    return grouped.sort_values("Velocity_Index", ascending=False)


def extract_keywords_and_frequencies(social_df: pd.DataFrame, min_word_len: int = 4) -> Dict[str, int]:
    """
    Extracts high-frequency fashion vocabulary from post captions, excluding stop words.
    """
    stop_words = {
        "this", "that", "with", "from", "have", "more", "your", "they", "will", "what",
        "their", "about", "there", "which", "when", "some", "them", "these", "very",
        "just", "into", "than", "look", "post", "fashionforecast", "http", "https", "season"
    }

    all_words = []
    for caption in social_df["Caption"]:
        # Clean text
        clean = re.sub(r"[^\w\s#]", "", caption.lower())
        words = clean.split()
        for w in words:
            if len(w) >= min_word_len and w not in stop_words and not w.startswith("#"):
                all_words.append(w)

    return dict(Counter(all_words).most_common(50))


def generate_fashion_wordcloud_image(social_df: pd.DataFrame, background_color: str = "#121212") -> plt.Figure:
    """
    Generates a high-resolution Matplotlib WordCloud figure styled for fashion editorial aesthetics.
    """
    word_freqs = extract_keywords_and_frequencies(social_df)
    if not word_freqs:
        word_freqs = {"tailoring": 10, "minimalist": 9, "cashmere": 8, "runway": 8, "vintage": 7}

    def luxury_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # Editorial color palette: gold, sage, cherry, lavender, cream
        colors = ["#E6C280", "#8A9A86", "#BA131A", "#B5A7D6", "#FFF1A8", "#E0E0E0", "#FFBE98"]
        return np.random.choice(colors)

    wc = WordCloud(
        width=900,
        height=450,
        background_color=background_color,
        color_func=luxury_color_func,
        max_words=60,
        font_path=None,
        prefer_horizontal=0.85,
        random_state=42
    ).generate_from_frequencies(word_freqs)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=background_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def get_sentiment_distribution(social_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns sentiment percentages and sentiment breakdown by aesthetic.
    """
    sentiment_counts = social_df["Sentiment_Label"].value_counts(normalize=True) * 100
    pos_pct = round(float(sentiment_counts.get("Positive", 0.0)), 1)
    neu_pct = round(float(sentiment_counts.get("Neutral", 0.0)), 1)
    neg_pct = round(float(sentiment_counts.get("Negative", 0.0)), 1)

    aesthetic_sentiment = social_df.groupby("Aesthetic_Mapped").agg(
        Avg_Sentiment=("Sentiment_Score", "mean"),
        Total_Posts=("Post_ID", "count")
    ).reset_index()
    aesthetic_sentiment["Avg_Sentiment"] = aesthetic_sentiment["Avg_Sentiment"].round(3)
    aesthetic_sentiment = aesthetic_sentiment.sort_values("Avg_Sentiment", ascending=False)

    return {
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "aesthetic_sentiment_df": aesthetic_sentiment
    }
