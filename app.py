"""
Automated Fashion Trend Prediction Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import sys

# Ensure src/ is accessible in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_engine import (
    get_cached_or_generate_dataset,
    generate_social_buzz_posts,
    generate_merchandise_sku_data,
    FASHION_CATEGORIES,
    AESTHETICS,
    TRENDING_COLORS,
    CITIES,
    DEMOGRAPHICS
)
from src.forecast_engine import (
    fit_trend_forecast,
    decompose_seasonality,
    compare_multiple_trends
)
from src.nlp_social import (
    analyze_hashtag_velocity,
    generate_fashion_wordcloud_image,
    get_sentiment_distribution
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
    run_trend_scenario_simulation,
    SUGGESTED_PROMPTS
)

# ---------------------------------------------------------
# Page Configuration & Modern Editorial Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VOGUE-AI | Automated Fashion Trend Prediction",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,800;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
        background: linear-gradient(135deg, #1A1A1A 0%, #5E503F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    @media (prefers-color-scheme: dark) {
        .main-title {
            background: linear-gradient(135deg, #F8F9FA 0%, #D2B48C 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .sub-title {
        font-size: 1.05rem;
        color: #7A7A7A;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(200, 200, 200, 0.15);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888888;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E6C280;
    }
    
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }

    .color-swatch-box {
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        color: #ffffff;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    
    .color-swatch-box:hover {
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data Initialization
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    df_trend = get_cached_or_generate_dataset()
    df_social = generate_social_buzz_posts()
    df_sku = generate_merchandise_sku_data()
    return df_trend, df_social, df_sku

df_trend, df_social, df_sku = load_all_data()


# ---------------------------------------------------------
# Sidebar Controls & Global Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🪐 **VOGUE-AI**")
    st.caption("Automated Fashion Trend Prediction & Analytics Engine v1.0")
    st.markdown("---")
    
    st.markdown("#### 🎯 **Global Filters**")
    selected_aesthetic = st.selectbox(
        "Focus Aesthetic",
        ["All Aesthetics"] + AESTHETICS,
        index=0
    )
    
    selected_category = st.selectbox(
        "Apparel Category",
        ["All Categories"] + FASHION_CATEGORIES,
        index=0
    )
    
    selected_city = st.selectbox(
        "Fashion Capital",
        ["Global (All Cities)"] + CITIES,
        index=0
    )

    st.markdown("---")
    st.markdown("#### ⚙️ **Model Parameters**")
    forecast_horizon = st.slider("Forecast Horizon (Months)", min_value=3, max_value=12, value=6, step=1)
    poly_degree = st.slider("Trend Fitting Degree", min_value=1, max_value=3, value=2, step=1)

    st.markdown("---")
    st.caption("Developed with Python, Streamlit, Scikit-Learn & Plotly.")


# Filter primary timeseries data based on sidebar
filtered_df = df_trend.copy()
if selected_aesthetic != "All Aesthetics":
    filtered_df = filtered_df[filtered_df["Aesthetic"] == selected_aesthetic]
if selected_category != "All Categories":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]
if selected_city != "Global (All Cities)":
    filtered_df = filtered_df[filtered_df["Primary_Region"] == selected_city]


# ---------------------------------------------------------
# Header & Navigation Tabs
# ---------------------------------------------------------
st.markdown('<div class="main-title">AUTOMATED FASHION TREND PREDICTOR</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicting Runway Trends, Aesthetic Lifecycles, Social Buzz, and Merchandising Demands</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🌟 Executive Overview",
    "📈 Time-Series Forecasting",
    "📱 Social Sentiment & Buzz",
    "🎨 Image Palette & Harmony",
    "🛍️ Inventory & Buying Plan",
    "🤖 AI Trend Copilot"
])


# =========================================================
# TAB 1: EXECUTIVE OVERVIEW
# =========================================================
with tabs[0]:
    st.markdown("### 📊 Macro Trend Intelligence & Market Momentum")
    
    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Calculate summary metrics
    latest_date = df_trend["Date"].max()
    curr_month_df = df_trend[df_trend["Date"] == latest_date]
    prev_month_df = df_trend[df_trend["Date"] == (latest_date - pd.DateOffset(months=1))]
    
    curr_avg_idx = curr_month_df["Search_Index"].mean()
    prev_avg_idx = prev_month_df["Search_Index"].mean() if len(prev_month_df) > 0 else curr_avg_idx
    idx_delta = round(((curr_avg_idx - prev_avg_idx) / prev_avg_idx) * 100, 1)

    top_aesthetic = curr_month_df.groupby("Aesthetic")["Search_Index"].mean().idxmax()
    runway_lead = curr_month_df.groupby("Category")["Runway_Share_Pct"].mean().idxmax()
    avg_sentiment = df_social["Sentiment_Score"].mean()

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Macro Trend Velocity</div>
            <div class="metric-value">{curr_avg_idx:.1f} / 100</div>
            <div class="metric-delta" style="color: {'#4CAF50' if idx_delta >= 0 else '#F44336'};">
                {'▲ +' if idx_delta >= 0 else '▼ '}{idx_delta}% MoM
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">#1 Dominant Aesthetic</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #FFF;">{top_aesthetic}</div>
            <div class="metric-delta" style="color: #4CAF50;">High Commercial Growth</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Runway Category Leader</div>
            <div class="metric-value" style="font-size: 1.35rem; color: #FFF;">{runway_lead}</div>
            <div class="metric-delta" style="color: #E6C280;">High Runway Share</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Consumer Sentiment</div>
            <div class="metric-value">{avg_sentiment * 100:.1f}%</div>
            <div class="metric-delta" style="color: #4CAF50;">▲ Bullish Consumer Affinity</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Main Visuals: Sunburst of Category/Aesthetic Share & Global Capitals Radar
    col_vis1, col_vis2 = st.columns([3, 2])

    with col_vis1:
        st.markdown("#### 💎 Aesthetic & Category Interest Distribution")
        sunburst_df = curr_month_df.groupby(["Category", "Aesthetic"])["Sales_Volume"].sum().reset_index()
        fig_sunburst = px.sunburst(
            sunburst_df,
            path=["Category", "Aesthetic"],
            values="Sales_Volume",
            color="Sales_Volume",
            color_continuous_scale="Viridis",
            title="Category vs Aesthetic Market Composition"
        )
        fig_sunburst.update_layout(margin=dict(t=30, l=0, r=0, b=0), height=420)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    with col_vis2:
        st.markdown("#### 🌍 Global Fashion Capital Momentum")
        city_metrics = df_trend[df_trend["Date"] == latest_date].groupby("Primary_Region")["Search_Index"].mean().reset_index()
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=city_metrics["Search_Index"].tolist() + [city_metrics["Search_Index"].iloc[0]],
            theta=city_metrics["Primary_Region"].tolist() + [city_metrics["Primary_Region"].iloc[0]],
            fill='toself',
            name='Capital Index',
            line_color='#E6C280',
            fillcolor='rgba(230, 194, 128, 0.25)'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(t=30, l=30, r=30, b=30),
            height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Color Palette of the Season
    st.markdown("#### 🎨 Key Palette Trends of the Season")
    color_cols = st.columns(len(TRENDING_COLORS[:5]))
    for idx, col_info in enumerate(TRENDING_COLORS[:5]):
        with color_cols[idx]:
            is_dark = col_info["family"] in ["Brown", "Red", "Blue", "Neutral"] and col_info["name"] != "Optic White"
            text_color = "#FFFFFF" if is_dark else "#1A1A1A"
            st.markdown(f"""
            <div class="color-swatch-box" style="background-color: {col_info['hex']}; color: {text_color};">
                <div style="font-size: 1.1rem; font-weight: 700;">{col_info['name']}</div>
                <div style="font-size: 0.85rem; opacity: 0.85;">{col_info['hex']}</div>
                <div style="font-size: 0.75rem; margin-top: 4px; opacity: 0.9;">{col_info['season']}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 2: TIME-SERIES FORECASTING
# =========================================================
with tabs[1]:
    st.markdown("### 📈 Predictive Fashion Time-Series & Seasonality Waves")
    st.caption("Machine Learning Polynomial & Seasonal Harmonic Models with 95% Confidence Intervals")

    # Run forecast on current filtered subset
    forecast_df, metrics = fit_trend_forecast(
        filtered_df,
        target_col="Search_Index",
        forecast_horizon_months=forecast_horizon,
        poly_degree=poly_degree
    )

    # Forecast Metrics Bar
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Current Search Score", f"{metrics['current_score']}/100")
    with m_col2:
        st.metric("Projected Score (+{}M)".format(forecast_horizon), f"{metrics['projected_score']}/100", f"{metrics['growth_pct']}%")
    with m_col3:
        st.metric("Expected Peak Period", metrics["peak_period"], f"Peak: {metrics['peak_score']}")
    with m_col4:
        st.metric("Trend Classification", metrics["trend_direction"], f"RMSE: {metrics['rmse']}")

    # Main Forecast Chart
    fig_fc = go.Figure()

    # Historical Line
    hist_part = forecast_df[~forecast_df["Historical"].isna()]
    fig_fc.add_trace(go.Scatter(
        x=hist_part["Date"],
        y=hist_part["Historical"],
        mode="lines+markers",
        name="Historical Search Volume",
        line=dict(color="#4A90E2", width=2.5),
        marker=dict(size=5)
    ))

    # Fitted line
    fig_fc.add_trace(go.Scatter(
        x=hist_part["Date"],
        y=hist_part["Fitted"],
        mode="lines",
        name="Polynomial Trendline",
        line=dict(color="#9B51E0", width=1.8, dash="dot")
    ))

    # Forecast Line
    future_part = forecast_df[forecast_df["Is_Forecast"]]
    fig_fc.add_trace(go.Scatter(
        x=future_part["Date"],
        y=future_part["Forecast"],
        mode="lines+markers",
        name=f"Forecast (+{forecast_horizon} Months)",
        line=dict(color="#E6C280", width=3),
        marker=dict(size=6, symbol="diamond")
    ))

    # Confidence Interval Shading
    fig_fc.add_trace(go.Scatter(
        x=future_part["Date"].tolist() + future_part["Date"].tolist()[::-1],
        y=future_part["Upper_CI"].tolist() + future_part["Lower_CI"].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(230, 194, 128, 0.18)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Band"
    ))

    fig_fc.update_layout(
        title=f"Trend Trajectory & Forecast Horizon: {selected_aesthetic} | {selected_category}",
        xaxis_title="Timeline",
        yaxis_title="Trend Interest Index (0 - 100)",
        hovermode="x unified",
        height=480,
        margin=dict(t=40, l=10, r=10, b=10)
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # Seasonality Decomposition & Multi-Trend Comparison
    col_seas, col_comp = st.columns([1, 1])

    with col_seas:
        st.markdown("#### 🍁 Annual Seasonality Impact (% Deviation)")
        season_df = decompose_seasonality(filtered_df)
        if not season_df.empty:
            fig_bar = px.bar(
                season_df,
                x="Month_Name",
                y="Impact_Pct",
                color="Impact_Pct",
                color_continuous_scale="Temps",
                title="Monthly Seasonal Effect on Demand"
            )
            fig_bar.update_layout(height=350, margin=dict(t=30, l=10, r=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Decomposition requires full annual series data.")

    with col_comp:
        st.markdown("#### ⚔️ Multi-Aesthetic Momentum Leaderboard")
        comparison_df = compare_multiple_trends(df_trend, group_col="Aesthetic", forecast_horizon_months=forecast_horizon)
        st.dataframe(
            comparison_df.style.background_gradient(subset=["Momentum Growth (%)"], cmap="YlGnBu"),
            use_container_width=True,
            height=320
        )


# =========================================================
# TAB 3: SOCIAL SENTIMENT & BUZZ
# =========================================================
with tabs[2]:
    st.markdown("### 📱 Social Media Intelligence, Virality & Sentiment Analysis")
    
    soc_col1, soc_col2 = st.columns([3, 2])
    
    with soc_col1:
        st.markdown("#### ⚡ Viral Hashtag Velocity Leaderboard")
        hashtag_df = analyze_hashtag_velocity(df_social)
        st.dataframe(
            hashtag_df.style.background_gradient(subset=["Velocity_Index", "Avg_Sentiment_Score"], cmap="OrRd"),
            use_container_width=True,
            height=340
        )
        
    with soc_col2:
        st.markdown("#### 💬 Sentiment Breakdown")
        sentiment_info = get_sentiment_distribution(df_social)
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[sentiment_info["positive_pct"], sentiment_info["neutral_pct"], sentiment_info["negative_pct"]],
            hole=0.55,
            marker_colors=["#4CAF50", "#FFC107", "#F44336"]
        )])
        fig_donut.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=340)
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # Fashion Wordcloud & Feed Explorer
    wc_col, feed_col = st.columns([1, 1])

    with wc_col:
        st.markdown("#### ☁️ Runway & Streetwear Keyword WordCloud")
        fig_wc = generate_fashion_wordcloud_image(df_social)
        st.pyplot(fig_wc)

    with feed_col:
        st.markdown("#### 📡 Real-Time Social Feed Scanner")
        selected_platform = st.selectbox("Filter Platform", ["All Platforms"] + list(df_social["Platform"].unique()))
        
        feed_filtered = df_social if selected_platform == "All Platforms" else df_social[df_social["Platform"] == selected_platform]
        
        for _, post in feed_filtered.head(4).iterrows():
            sent_color = "#4CAF50" if post["Sentiment_Label"] == "Positive" else ("#FFC107" if post["Sentiment_Label"] == "Neutral" else "#F44336")
            st.markdown(f"""
            <div style="border-left: 3px solid {sent_color}; padding-left: 12px; margin-bottom: 12px; background: rgba(255,255,255,0.02); padding-top: 6px; padding-bottom: 6px;">
                <div style="font-size: 0.8rem; color: #888;"><b>{post['Platform']}</b> • {post['Influencer_Tier']} • {post['City']} • {post['Date']}</div>
                <div style="font-size: 0.92rem; margin-top: 3px;">{post['Caption']}</div>
                <div style="font-size: 0.8rem; color: #E6C280; margin-top: 4px;">❤️ {post['Likes']:,} | 🔄 {post['Shares']:,} | Sentiment: {post['Sentiment_Score']}</div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# TAB 4: IMAGE PALETTE & COLOR HARMONY
# =========================================================
with tabs[3]:
    st.markdown("### 🎨 Computer Vision Color Palette & Harmony Extractor")
    st.caption("Upload any runway photo, moodboard, or garment image to extract exact HEX colors, RGB values, and aesthetic harmony.")

    img_input_mode = st.radio(
        "Select Image Source:",
        ["Choose a Sample Moodboard", "Upload Custom Garment / Runway Photo"],
        horizontal=True
    )

    garment_img = None

    if img_input_mode == "Choose a Sample Moodboard":
        sample_choice = st.selectbox(
            "Select Sample Swatch:",
            ["Quiet Luxury (Sage, Camel, Espresso)", "Cherry Red Streetwear (Red, Chrome, Obsidian)", "Spring Pastel (Butter Yellow, Lavender, Cobalt)"]
        )
        if "Quiet Luxury" in sample_choice:
            garment_img = create_sample_fashion_image("Quiet Luxury")
        elif "Cherry Red" in sample_choice:
            garment_img = create_sample_fashion_image("Cherry Red Bold")
        else:
            garment_img = create_sample_fashion_image("Pastel")
    else:
        uploaded_file = st.file_uploader("Upload Image File (JPG, PNG, WEBP)", type=["jpg", "png", "jpeg", "webp"])
        if uploaded_file is not None:
            garment_img = Image.open(uploaded_file)
        else:
            st.info("👆 Upload an image above or switch to sample swatches.")

    if garment_img is not None:
        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.image(garment_img, caption="Analyzed Garment / Swatch", use_column_width=True)

        with c_right:
            k_clusters = st.slider("Number of Dominant Colors to Extract (K-Means)", min_value=3, max_value=8, value=5)
            extracted = extract_dominant_colors_from_image(garment_img, num_colors=k_clusters)
            harmony = analyze_color_harmony(extracted)

            st.markdown(f"#### 🏷️ Harmony Classification: **{harmony['harmony_type']}**")
            st.caption(harmony["description"])

            st.markdown("##### Extracted Color Swatches:")
            swatch_cols = st.columns(len(extracted))
            for i, col_data in enumerate(extracted):
                with swatch_cols[i]:
                    rgb_val = col_data["rgb"]
                    is_dark = (rgb_val[0]*0.299 + rgb_val[1]*0.587 + rgb_val[2]*0.114) < 140
                    t_col = "#FFFFFF" if is_dark else "#000000"
                    
                    st.markdown(f"""
                    <div style="background-color: {col_data['hex']}; color: {t_col}; border-radius: 8px; padding: 10px; text-align: center; font-size: 0.8rem; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
                        <b>{col_data['percentage']}%</b><br>
                        {col_data['hex']}<br>
                        <span style="font-size: 0.7rem;">{col_data['name']}</span>
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================
# TAB 5: INVENTORY & BUYING PLAN
# =========================================================
with tabs[4]:
    st.markdown("### 🛍️ Retail Merchandising & Inventory Planning Engine")
    st.caption("Translating predicted trend velocity into SKU-level procurement recommendations and markdown mitigation.")

    # Calculate Merchandising Strategy
    strat_df = calculate_merchandise_strategy(df_sku)
    budget_summary = generate_budget_reallocation_summary(df_sku)

    # Budget re-allocation banner
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.metric("Top Investment Aesthetic", budget_summary["top_growth_aesthetic"], "Scale Production")
    with b_col2:
        st.metric("Highest Markdown Risk", budget_summary["most_at_risk_aesthetic"], "Tighten Reorders")
    with b_col3:
        st.metric("Average Gross Margin", f"{strat_df['Gross_Margin_Pct'].mean():.1f}%", "Healthy Margin")

    st.markdown("---")

    # Risk vs Momentum Matrix Chart
    st.markdown("#### 🎯 SKU Risk Matrix (Forecast Growth % vs Gross Margin %)")
    fig_bubble = px.scatter(
        strat_df,
        x="Forecast_Growth_Pct",
        y="Gross_Margin_Pct",
        size="Current_Stock_Units",
        color="Risk_Tier",
        hover_name="Item_Name",
        hover_data=["Category", "Aesthetic", "Procurement_Recommendation"],
        color_discrete_map={
            "Low (High Demand Surge)": "#4CAF50",
            "Low-Medium (Stable Growth)": "#2196F3",
            "Medium (Plateauing Trend)": "#FF9800",
            "High (Declining Momentum)": "#F44336"
        },
        title="SKU Positioning: Growth Momentum vs Profitability"
    )
    fig_bubble.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_bubble.update_layout(height=420, margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig_bubble, use_container_width=True)

    # SKU Table
    st.markdown("#### 📋 Recommended Procurement Actions")
    # Safely apply styling across pandas versions
    styler = strat_df[[
        "SKU_Code", "Item_Name", "Aesthetic", "Color_Palette", 
        "Current_Stock_Units", "Forecast_Growth_Pct", "Recommended_Action", "Risk_Tier"
    ]].style
    if hasattr(styler, "map"):
        styled_table = styler.map(
            lambda val: "color: #4CAF50; font-weight: bold;" if "Increase" in str(val) else ("color: #F44336; font-weight: bold;" if "Markdown" in str(val) else ""),
            subset=["Recommended_Action"]
        )
    else:
        styled_table = styler.applymap(
            lambda val: "color: #4CAF50; font-weight: bold;" if "Increase" in str(val) else ("color: #F44336; font-weight: bold;" if "Markdown" in str(val) else ""),
            subset=["Recommended_Action"]
        )
    st.dataframe(
        styled_table,
        use_container_width=True,
        height=350
    )

    # CSV Download Button
    csv_bytes = strat_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Merchandising Plan (CSV)",
        data=csv_bytes,
        file_name="fashion_merchandise_buying_plan_2026.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 6: AI TREND COPILOT & SIMULATOR
# =========================================================
with tabs[5]:
    st.markdown("### 🤖 AI Fashion Trend Copilot & Scenario Simulator")
    
    sim_col, chat_col = st.columns([1, 1])

    with sim_col:
        st.markdown("#### 🧪 What-If Trend Shock Simulator")
        st.caption("Simulate unexpected market demand shifts and evaluate unit & revenue impact.")
        
        sim_aesthetic = st.selectbox("Select Target Aesthetic for Shock", AESTHETICS, index=0)
        sim_shock = st.slider("Simulated Demand Shock (%)", min_value=-50, max_value=100, value=25, step=5)

        if st.button("⚡ Run Scenario Simulation", type="primary"):
            sim_res = run_trend_scenario_simulation(sim_aesthetic, sim_shock, df_trend)
            st.success(f"Simulation Complete for **{sim_res['aesthetic']}**")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Simulated Search Index", f"{sim_res['simulated_search_index']}", f"{sim_shock}%")
            with c2:
                st.metric("Estimated Monthly Unit Delta", f"{sim_res['estimated_monthly_units_delta']:+d} units")
            
            st.metric("Projected Monthly Revenue Impact", f"${sim_res['projected_revenue_impact_usd']:+,.2f} USD")

    with chat_col:
        st.markdown("#### 💬 Strategic Advisory Assistant")
        st.caption("Ask queries about emerging styles, runway insights, color pairings, and stocking decisions.")

        selected_prompt = st.selectbox("Quick-Select Strategic Prompt:", ["-- Custom Question --"] + SUGGESTED_PROMPTS)
        
        if selected_prompt != "-- Custom Question --":
            user_query = selected_prompt
        else:
            user_query = st.text_input("Enter your fashion intelligence question:", "What are the top 3 highest momentum aesthetics right now?")

        if st.button("Consult Copilot", type="secondary"):
            with st.spinner("Analyzing fashion knowledge base & current forecast indices..."):
                ans = ask_fashion_copilot(user_query, df_trend, df_sku)
                st.markdown(ans["response"])

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #777; font-size: 0.85rem;'>"
    "VOGUE-AI • Automated Fashion Trend Forecasting & Intelligence System • Built with Streamlit & Python"
    "</div>",
    unsafe_allow_html=True
)
