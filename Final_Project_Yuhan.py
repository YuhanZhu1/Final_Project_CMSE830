# Final_Project_Yuhan.py
# CMSE 830 Final Project - World Cup Prediction
# Deployment-hardened for Streamlit Cloud (Python 3.9, streamlit==1.12.2)
#
# Key fixes:
# - Streamlit cache compatibility (st.cache_data not available in 1.12.2)
# - Robust anchor_date handling (always pd.Timestamp)
# - Safe file loading with clearer messages
# - Numeric-only correlation heatmap with empty checks
# - Defensive merges / data checks to avoid "redacted" crashes

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="CMSE 830 Final Project - World Cup Prediction", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# Streamlit cache compatibility: Streamlit 1.12.2 uses st.cache, newer uses st.cache_data
cache_data = getattr(st, "cache_data", st.cache)

@cache_data(show_spinner=False)
def load_csv(filename: str) -> pd.DataFrame:
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {filename}. Expected at: {path}")
    return pd.read_csv(path)

def safe_image(path: str, width: int = None):
    p = BASE_DIR / path
    if p.exists():
        st.image(str(p), width=width)
    else:
        st.warning(f"Missing image file: {path}")

def normalize_country_names(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # Keep exactly what you had, just wrapped safely
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("IR Iran", "Iran", regex=False)
            .str.replace("Korea Republic", "South Korea", regex=False)
            .str.replace("USA", "United States", regex=False)
        )
    return df

# -----------------------
# Tabs
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Data EDA", "Training & Testing", "Summary"])

# -----------------------
# Introduction
# -----------------------
with tab1:
    st.markdown(
        """
#### CMSE830 Final Project 🎉
## ⚽️ 2022 World Cup Prediction ⚽️
"""
    )
    safe_image("960x0.jpg", width=900)

    st.markdown(
        """
**Goal:** Predict match outcomes for the 2022 World Cup using historical results and FIFA rankings.

**Datasets**
1) FIFA World Ranking (1992–2022, Oct 6)  
2) International match results (1872–2022, Dec 6)

**Models (trained offline / evaluated in notebook)**
BernoulliNB, DecisionTreeClassifier, GradientBoostingClassifier, GaussianNB, LogisticRegression, MLPClassifier, RandomForestClassifier
"""
    )

    intro_summary = st.selectbox(label="", options=("", "Just Tell Me The Result 🤡"))
    if intro_summary == "Just Tell Me The Result 🤡":
        st.markdown(
            """
Quarter Finals:
- Croatia vs Brazil: Brazil Win  
- Netherlands vs Argentina: Netherlands Win  
- Morocco vs Portugal: Portugal Win  
- England vs France: France Win  

Semi-finals:
- Brazil vs Netherlands: Brazil Win  
- France vs Portugal: Portugal Win  

3rd:
- Netherlands vs France: Netherlands Win  

Final:
- Brazil vs Portugal: Brazil Win
"""
        )

# -----------------------
# Exploratory Data Analysis
# -----------------------
with tab2:
    st.header("Exploratory Data Analysis 🥸")

    # 1) FIFA ranking
    st.subheader("1) FIFA World Ranking (since 2018)")
    try:
        rank = load_csv("fifa_ranking-2022-10-06.csv")
    except Exception as e:
        st.error(f"Failed to load fifa_ranking-2022-10-06.csv: {e}")
        st.stop()

    if "rank_date" not in rank.columns:
        st.error("Expected column 'rank_date' not found in fifa ranking file.")
        st.stop()

    rank["rank_date"] = pd.to_datetime(rank["rank_date"], errors="coerce")
    rank = rank.dropna(subset=["rank_date"]).copy()
    rank = rank[rank["rank_date"] >= "2018-08-01"].reset_index(drop=True)

    # Normalize names for consistency with results
    rank = normalize_country_names(rank, "country_full")

    st.write(rank.head(), rank.shape)

    # Pick top 10 teams based on an anchor date (closest available >= 2018-08-16)
    anchor = pd.Timestamp("2018-08-16")
    available_dates = pd.to_datetime(rank["rank_date"].dropna().unique(), errors="coerce")
    available_dates = pd.DatetimeIndex(available_dates).dropna().sort_values()

    if len(available_dates) == 0:
        st.error("No valid ranking dates after parsing. Please check fifa_ranking CSV.")
        st.stop()

    # earliest date >= anchor; if none, last available date
    anchor_date = available_dates[available_dates >= anchor].min()
    if pd.isna(anchor_date):
        anchor_date = available_dates.max()

    anchor_date = pd.Timestamp(anchor_date)  # guarantee Timestamp
    st.caption(f"Top 10 selection date: {anchor_date.date()}")

    # Make sure required columns exist
    required_cols = {"country_full", "total_points"}
    missing = required_cols - set(rank.columns)
    if missing:
        st.error(f"Missing required columns in ranking data: {missing}")
        st.stop()

    top10 = (
        rank[rank["rank_date"] == anchor_date]
        .sort_values("total_points", ascending=False)
        .head(10)["country_full"]
        .tolist()
    )

    if len(top10) == 0:
        st.warning("Top 10 team selection returned empty. Check anchor_date filtering.")
    else:
        # Plot ranking evolution without pivot (fast)
        best = rank[rank["country_full"].isin(top10)].copy()
        best = best.sort_values(["country_full", "rank_date"])

        fig = go.Figure()
        for team in top10:
            tdf = best[best["country_full"] == team]
            fig.add_trace(
                go.Scatter(
                    x=tdf["rank_date"],
                    y=tdf["total_points"],
                    name=team,
                    mode="lines",
                )
            )

        fig.update_layout(
            title="FIFA Ranking Evolution (Total Points) for Top 10 Teams",
            xaxis_title="Date",
            yaxis_title="Total Points",
            legend_title="Team",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2) Match results
    st.subheader("2) Match Results (since 2018)")
    try:
        results = load_csv("results.csv")
    except Exception as e:
        st.error(f"Failed to load results.csv: {e}")
        st.stop()

    if "date" not in results.columns:
        st.error("Expected column 'date' not found in results.csv.")
        st.stop()

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date"]).copy()
    results = results[results["date"] >= "2018-08-01"].reset_index(drop=True)

    # Normalize team names to match ranking
    results = normalize_country_names(results, "home_team")
    results = normalize_country_names(results, "away_team")

    st.write(results.head(), results.shape)

    # 3) Merge ranking features into match results
    st.subheader("3) Merge ranking features into match results")

    needed_rank_cols = ["rank_date", "country_full", "total_points", "previous_points", "rank", "rank_change"]
    missing_rank_cols = [c for c in needed_rank_cols if c not in rank.columns]
    if missing_rank_cols:
        st.error(f"Ranking file missing columns needed for merge/features: {missing_rank_cols}")
        st.stop()

    r = rank[needed_rank_cols].copy()
    r = r.sort_values(["country_full", "rank_date"])

    # Resample to daily per country, forward fill
    r = (
        r.set_index("rank_date")
        .groupby("country_full", group_keys=False)
        .resample("D")
        .first()
        .ffill()
        .reset_index()
    )

    # Merge home
    try:
        df_merge = results.merge(
            r,
            left_on=["date", "home_team"],
            right_on=["rank_date", "country_full"],
            how="left",
        ).drop(["rank_date", "country_full"], axis=1)
    except Exception as e:
        st.error(f"Home merge failed: {e}")
        st.stop()

    # Merge away
    try:
        df = df_merge.merge(
            r,
            left_on=["date", "away_team"],
            right_on=["rank_date", "country_full"],
            how="left",
            suffixes=("_home", "_away"),
        ).drop(["rank_date", "country_full"], axis=1)
    except Exception as e:
        st.error(f"Away merge failed: {e}")
        st.stop()

    st.write(df.head(), df.shape)

    # Correlation heatmap
    st.subheader("Correlation heatmap (numeric features only)")
    num_df = df.select_dtypes(include=[np.number]).copy()

    if num_df.shape[1] < 2:
        st.info("Not enough numeric columns to compute correlations.")
    else:
        corr = num_df.corr()

        fig_hm, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, cmap="viridis", ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)")
        st.pyplot(fig_hm)

# -----------------------
# Training & Testing (display precomputed results)
# -----------------------
with tab3:
    st.header("Training & Testing")

    try:
        data = load_csv("model_data.csv")
    except Exception as e:
        st.error(f"Failed to load model_data.csv: {e}")
        st.stop()

    st.write(data.describe())
    st.write("Shape:", data.shape)

    st.markdown("### Which ML Model would you like to have a look at?")
    model = st.selectbox(
        label="",
        options=(
            "",
            "None of them",
            "BernoulliNB",
            "DecisionTreeClassifier",
            "GradientBoostingClassifier",
            "GaussianNB",
            "LogisticRegression",
            "MLPClassifier",
            "RandomForestClassifier",
        ),
    )

    if model == "None of them":
        safe_image("17c.jpg", width=900)

    if model == "BernoulliNB":
        safe_image("BNB.png", width=900)
        safe_image("BNB_CV.png", width=900)

    if model == "DecisionTreeClassifier":
        safe_image("DecisionTreeClassifier.png", width=900)
        safe_image("DecisionTreeClassifier_CV.png", width=900)

    if model == "GradientBoostingClassifier":
        safe_image("GradientBoostingClassifier.png", width=900)
        safe_image("GradientBoostingClassifier_CV.png", width=900)

    if model == "GaussianNB":
        safe_image("GaussianNB.png", width=900)
        safe_image("GaussianNB_CV.png", width=900)

    if model == "LogisticRegression":
        safe_image("LogisticRegression.png", width=900)
        safe_image("LogisticRegression_CV.png", width=900)

    if model == "MLPClassifier":
        safe_image("MLPClassifier.png", width=900)
        safe_image("MLPClassifier_CV.png", width=900)

    if model == "RandomForestClassifier":
        safe_image("RandomForestClassifier.png", width=900)
        safe_image("RandomForestClassifier_CV.png", width=900)

# -----------------------
# Summary
# -----------------------
with tab4:
    safe_image("ThisMorning.png", width=900)

    summary = st.selectbox(label="", options=("", "Here's the prediction"))
    if summary == "Here's the prediction":
        st.markdown(
            """
Quarter Finals:
- Croatia vs Brazil: Brazil Win  
- Netherlands vs Argentina: Netherlands Win  
- Morocco vs Portugal: Portugal Win  
- England vs France: France Win  

Top 4:
Semi-finals:
- Brazil vs Netherlands: Brazil Win  
- France vs Portugal: Portugal Win  

3rd:
- Netherlands vs France: Netherlands Win  

Final:
- Brazil vs Portugal: Brazil Win
"""
        )

    st.markdown(
        """
#### Things I learned
- Cleaning and aligning real-world data took the majority of the effort.
- Different classifiers behave very differently on the same dataset.
- Sports outcomes have inherent uncertainty, so probabilities and evaluation matter.

#### Reflection
- Real data integration (country name mismatches, date alignment, missing values) was more complex than expected.
- A next step would be adding an interactive matchup tool: pick two teams and output win probabilities.

#### References
- FIFA ranking dataset: Kaggle (FIFA rankings)  
- Match results dataset: Kaggle (international football results)  
- Cleaning inspiration: Kaggle notebook on FIFA 2022 prediction  
- Photo credit: FIFA/Getty Images

## Thank you for listening
"""
    )

