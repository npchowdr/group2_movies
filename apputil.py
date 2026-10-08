import plotly.express as px
import pandas as pd

# update/add code below ...


"""
Helper functions for the genre risk/reward app.
Pure data / plotting logic -- no Streamlit calls -- so it can be imported and tested anywhere.
Mirrors the conventions of the team's budget-distribution app (same YEAR_MIN/YEAR_MAX,
GENRES list, PALETTE, and fmt_money helper) so the pages feel consistent side by side.
"""

import numpy as np
import plotly.graph_objects as go

YEAR_MIN, YEAR_MAX = 2010, 2018  # years with a meaningful sample (~120-200 films each)
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]  # "TV Movie" omitted: only 3 films in range
DEFAULT_GENRE = "Adventure"
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#EECA3B", "#BAB0AC", "#1F77B4", "#D62728",
]


# ---------------------------------------------------------------- data
def load_data(source) -> pd.DataFrame:
    """Read the CSV and keep 2010-2018 films with a positive budget."""
    df = pd.read_csv(source)
    df = df[df["year"].between(YEAR_MIN, YEAR_MAX)].copy()
    df = df[df["production_budget"] > 0]
    return df


def filter_data(df: pd.DataFrame, years: tuple[int, int], exclude_no_bo: bool) -> pd.DataFrame:
    """Apply the year range and optional no-box-office exclusion."""
    out = df[df["year"].between(*years)]
    if exclude_no_bo:
        out = out[~out["no_reported_box_office"]]
    return out


def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"


def genre_color(genre: str, genres: list[str]) -> str:
    return PALETTE[genres.index(genre) % len(PALETTE)]


# ---------------------------------------------------------------- Q3: risk/reward
def get_genre_risk_reward(df: pd.DataFrame, genres: list[str] = GENRES, min_films: int = 10) -> pd.DataFrame:
    """
    Per-genre table of mean ROI, ROI std dev, and coefficient of variation
    (std/mean -- lower = more consistent returns, higher = more boom-or-bust).
    Genres below min_films are dropped (too few films for a reliable estimate).
    """
    rows = []
    for g in genres:
        subset = df.loc[df[g] == 1, "roi"].dropna()
        if len(subset) >= min_films:
            mean_roi = subset.mean()
            std_roi = subset.std()
            cv = std_roi / mean_roi if mean_roi != 0 else float("nan")
            rows.append({
                "genre": g, "count": len(subset),
                "mean_roi": mean_roi, "std_roi": std_roi,
                "coefficient_of_variation": cv,
            })
    return pd.DataFrame(rows).sort_values("coefficient_of_variation").reset_index(drop=True)


def risk_reward_figure(risk: pd.DataFrame) -> go.Figure:
    """
    Bubble chart: every genre by mean ROI (x) vs. coefficient of variation (y).
    Bubble size = film count, color = volatility. This is the Q3 answer made explorable.

    Only genres that stand out from the pack get a permanent text label --
    labeling all of them makes the dense low-volatility cluster unreadable.
    Every genre is still hoverable (hover_name) even without a visible label.
    """
    cv = risk["coefficient_of_variation"]
    roi = risk["mean_roi"]
    cv_z = (cv - cv.mean()) / cv.std(ddof=0)
    roi_z = (roi - roi.mean()) / roi.std(ddof=0)
    is_outlier = (cv_z.abs() > 0.75) | (roi_z.abs() > 1.25)
    risk = risk.assign(label=risk["genre"].where(is_outlier, ""))

    fig = px.scatter(
        risk, x="mean_roi", y="coefficient_of_variation", size="count",
        color="coefficient_of_variation", color_continuous_scale="RdYlGn_r",
        text="label", hover_name="genre",
        hover_data={"mean_roi": ":.2f", "coefficient_of_variation": ":.2f", "count": True,
                    "label": False},
        size_max=55,
        labels={"mean_roi": "Average ROI (return per $1 spent)",
                "coefficient_of_variation": "Volatility (coefficient of variation)",
                "count": "Number of films"},
    )
    fig.update_traces(textposition="top center", textfont=dict(size=12))
    fig.add_hline(
        y=risk["coefficient_of_variation"].median(),
        line_dash="dot", line_color="gray",
        annotation_text="median volatility", annotation_position="bottom right",
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10),
                      coloraxis_colorbar_title="Volatility")
    return fig


# ---------------------------------------------------------------- Q1/Q3: genre extremes
def genre_extremes(df: pd.DataFrame, genre: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Films in `genre`, plus the biggest-budget film and the lowest-ROI film in it."""
    subset = df[df[genre] == 1].dropna(subset=["roi", "production_budget"])
    biggest = subset.loc[subset["production_budget"].idxmax()]
    worst = subset.loc[subset["roi"].idxmin()]
    return subset, biggest, worst


def extremes_figure(subset: pd.DataFrame, biggest: pd.Series, worst: pd.Series, genre: str) -> go.Figure:
    """
    Scatter of production budget (log scale) vs. ROI for every film in `genre`,
    with the biggest-budget film and lowest-ROI film starred and labeled --
    a blockbuster-vs-bust callout within one genre.
    """
    fig = px.scatter(
        subset, x="production_budget", y="roi", log_x=True,
        hover_name="movie",
        hover_data={"year": True, "production_budget": ":$,.0f", "roi": ":.2f"},
        opacity=0.55,
        labels={"production_budget": "Production Budget ($, log scale)", "roi": "ROI"},
    )
    fig.update_traces(marker=dict(color="#BAB0AC", size=8))

    fig.add_trace(go.Scatter(
        x=[biggest["production_budget"]], y=[biggest["roi"]], mode="markers+text",
        marker=dict(symbol="star", size=22, color="#4C78A8", line=dict(width=1, color="black")),
        text=[f"{biggest['movie']} ({int(biggest['year'])})<br>Biggest blockbuster"],
        textposition="top center", name="Biggest blockbuster",
        hovertemplate=(f"<b>{biggest['movie']}</b><br>Budget: ${biggest['production_budget']:,.0f}"
                       f"<br>ROI: {biggest['roi']:.2f}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=[worst["production_budget"]], y=[worst["roi"]], mode="markers+text",
        marker=dict(symbol="star", size=22, color="#E45756", line=dict(width=1, color="black")),
        text=[f"{worst['movie']} ({int(worst['year'])})<br>Lowest ROI"],
        textposition="bottom center", name="Lowest ROI",
        hovertemplate=(f"<b>{worst['movie']}</b><br>Budget: ${worst['production_budget']:,.0f}"
                       f"<br>ROI: {worst['roi']:.2f}<extra></extra>"),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black",
                  annotation_text="break-even", annotation_position="bottom right")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), showlegend=True)
    return fig


# ---------------------------------------------------------------- stats tab
def risk_reward_table(risk: pd.DataFrame) -> pd.DataFrame:
    """risk_reward table formatted for st.dataframe display (renamed, rounded columns)."""
    out = risk.rename(columns={
        "genre": "Genre", "count": "Films", "mean_roi": "Avg ROI",
        "std_roi": "ROI Std Dev", "coefficient_of_variation": "Volatility (CV)",
    })
    return out
