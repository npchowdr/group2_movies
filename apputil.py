"""
Helper functions for the movie budget distribution app.
Pure data / plotting logic — no Streamlit calls — so it can be imported and tested anywhere.
"""


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

YEAR_MIN, YEAR_MAX = 2010, 2018  # years with a meaningful sample (~120-200 films each)
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]  # "TV Movie" omitted: only 3 films in range
DEFAULT_GENRE = "Adventure"
DEFAULT_GENRES = ["Action", "Animation", "Comedy", "Drama", "Horror"]
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#EECA3B", "#BAB0AC", "#1F77B4", "#D62728",
]

REF_COLOR = "#888888"


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


def genre_budgets(df: pd.DataFrame, genres: list[str], min_films: int = 3) -> dict[str, np.ndarray]:
    """Map each genre to an array of its films' budgets (genres with too few films dropped)."""
    groups = {g: df.loc[df[g] == 1, "production_budget"].to_numpy(float) for g in genres}
    return {g: v for g, v in groups.items() if len(v) >= min_films}


def genre_color(genre: str, genres: list[str]) -> str:
    return PALETTE[genres.index(genre) % len(PALETTE)]


# ---------------------------------------------------------------- helpers
def fmt_money(v: float) -> str:
    if v >= 1e9:
        return f"${v/1e9:.2f}B"
    if v >= 1e6:
        return f"${v/1e6:.1f}M"
    if v >= 1e3:
        return f"${v/1e3:.0f}K"
    return f"${v:,.0f}"

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

def hex_to_rgba(color: str, alpha: float) -> str:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def gaussian_kde(samples: np.ndarray, grid: np.ndarray, bw_adjust: float = 1.0,
                 reflect_at: float | None = None) -> np.ndarray:
    """Gaussian KDE with Scott's rule bandwidth (numpy only).
    Optional reflection at a lower boundary so the curve doesn't leak below it."""
    n = len(samples)
    if n < 2:
        return np.zeros_like(grid)
    bw = max(1.06 * samples.std(ddof=1) * n ** (-1 / 5) * bw_adjust, 1e-9)
    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1)
    if reflect_at is not None:
        z_ref = (grid[:, None] - (2 * reflect_at - samples)[None, :]) / bw
        dens += np.exp(-0.5 * z_ref**2).sum(axis=1)
    return dens / (n * bw * np.sqrt(2 * np.pi))


def _budget_grid(all_budgets: np.ndarray, log_scale: bool, points: int = 400):
    """Return (grid in transformed space, grid in dollars, transform fn, reflect boundary)."""
    if log_scale:
        lo, hi = np.log10(all_budgets.min()), np.log10(all_budgets.max())
        grid_t = np.linspace(lo - 0.3, hi + 0.2, points)
        return grid_t, 10 ** grid_t, np.log10, None
    grid_t = np.linspace(0, all_budgets.max() * 1.05, points)
    return grid_t, grid_t, (lambda a: a), 0.0


# ---------------------------------------------------------------- figures
def distribution_figure(
    groups: dict[str, np.ndarray],
    all_budgets: np.ndarray,
    *,
    log_scale: bool = True,
    style: str = "Density curve",       # "Density curve" | "Histogram" | "Both"
    scaled: bool = False,               # True: height scaled by number of films
    bw_adjust: float = 1.0,
    show_medians: bool = True,
    show_all: bool = True,
) -> go.Figure:
    """Overlaid density curves / histograms of budgets for each genre."""
    grid_t, grid_x, transform, reflect = _budget_grid(all_budgets, log_scale)
    genre_list = list(groups)
    series = list(groups.items())
    if show_all:
        series = [("All films", all_budgets)] + series

    fig = go.Figure()
    for name, vals in series:
        is_ref = name == "All films"
        color = REF_COLOR if is_ref else genre_color(name, genre_list)
        weight = len(vals) if scaled else 1.0

        if style in ("Histogram", "Both"):
            counts, edges = np.histogram(transform(vals), bins=30, range=(grid_t[0], grid_t[-1]))
            width = edges[1] - edges[0]
            heights = counts / (len(vals) * width) * weight  # same units as the density curve
            centers = (edges[:-1] + edges[1:]) / 2
            fig.add_trace(go.Bar(
                x=10 ** centers if log_scale else centers, y=heights,
                name=f"{name} (hist)", legendgroup=name, marker_color=color,
                opacity=0.25 if style == "Both" else 0.55, showlegend=style == "Histogram",
                width=(10 ** edges[1:] - 10 ** edges[:-1]) if log_scale else width,
                customdata=counts,
                hovertemplate=f"<b>{name}</b><br>~%{{x:$,.3s}}<br>%{{customdata}} films<extra></extra>",
            ))

        if style in ("Density curve", "Both"):
            dens = gaussian_kde(transform(vals), grid_t, bw_adjust, reflect) * weight
            fig.add_trace(go.Scatter(
                x=grid_x, y=dens, mode="lines", name=f"{name} (n={len(vals)})", legendgroup=name,
                line=dict(color=color, width=2 if is_ref else 3, dash="dot" if is_ref else "solid"),
                fill=None if is_ref else "tozeroy", fillcolor=hex_to_rgba(color, 0.08),
                hovertemplate=f"<b>{name}</b><br>Budget: %{{x:$,.3s}}<extra></extra>",
            ))

        if show_medians and not is_ref:
            fig.add_vline(x=float(np.median(vals)), line=dict(color=color, width=1.5, dash="dash"),
                          opacity=0.8)

    fig.update_layout(
        height=560, barmode="overlay", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_title=("Density × films" if scaled else "Density") + (" (per log₁₀ $)" if log_scale else ""),
        xaxis_title="Production budget",
    )
    fig.update_xaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat="~s")
    fig.update_yaxes(showticklabels=scaled)  # normalized density values aren't meaningful to read
    return fig


def violin_figure(df: pd.DataFrame, groups: dict[str, np.ndarray], log_scale: bool = True) -> go.Figure:
    """Half-violins per genre, sorted by median, with hoverable film titles."""
    genre_list = list(groups)
    order = sorted(groups, key=lambda g: np.median(groups[g]))
    fig = go.Figure()
    for g in order:
        color = genre_color(g, genre_list)
        fig.add_trace(go.Violin(
            x=groups[g], name=f"{g} (n={len(groups[g])})", orientation="h", side="positive",
            line_color=color, fillcolor=color, opacity=0.6, width=1.8,
            points="all" if len(groups[g]) < 400 else False, pointpos=-0.15, jitter=0.1,
            marker=dict(size=3), box_visible=True, hoveron="points",
            text=df.loc[df[g] == 1, "movie"],
            hovertemplate="<b>%{text}</b><br>%{x:$,.3s}<extra></extra>",
        ))
    fig.update_layout(height=max(320, 90 * len(order) + 80), showlegend=False,
                      margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Production budget")
    fig.update_xaxes(type="log" if log_scale else "linear", tickprefix="$", tickformat="~s")
    return fig


# ---------------------------------------------------------------- stats
MONEY_COLS = ["Median", "Mean", "25th pct", "75th pct", "Max"]


def summary_stats(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Per-genre budget summary, sorted by median (highest first)."""
    rows = [{
        "Genre": g, "Films": len(v),
        "Median": np.median(v), "Mean": v.mean(),
        "25th pct": np.percentile(v, 25), "75th pct": np.percentile(v, 75),
        "Max": v.max(),
        "% under $10M": (v < 10e6).mean() * 100,
        "% over $100M": (v > 100e6).mean() * 100,
    } for g, v in groups.items()]
    return pd.DataFrame(rows).sort_values("Median", ascending=False).reset_index(drop=True)


def stats_for_display(stats: pd.DataFrame) -> pd.DataFrame:
    """Copy of summary_stats with dollar columns converted to millions (for display formatting)."""
    out = stats.copy()
    out[MONEY_COLS] = out[MONEY_COLS] / 1e6
    return out

def mean_std_table(df, genres=GENRES):
    rows = []
    for g in genres:
        v = df.loc[df[g] == 1, "production_budget"].to_numpy(float)
        if len(v) < 2:
            continue
        rows.append({"Genre": g, "Films": len(v),
                     "Average budget": v.mean() / 1e6,
                     "Std deviation": v.std(ddof=1) / 1e6})
    return pd.DataFrame(rows).sort_values("Average budget", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- Q1: low-budget breakouts
ALL_GENRES = "All genres"
BUDGET_STEPS = [1e6, 2e6, 3e6, 5e6, 7e6, 9e6, 12e6, 15e6, 20e6, 30e6, 50e6, 100e6, 425e6]


def breakout_tag(df: pd.DataFrame, max_budget: float, genre: str, multiple: float) -> pd.DataFrame:
    """Films with reported box office, plus three columns: `multiple` (gross / budget),
    `selected` (fits the budget cap + genre) and `breakout` (grossed >= multiple x budget)."""
    out = df[~df["no_reported_box_office"]].copy()   # a $0 gross would count as a flop
    out["multiple"] = out["worldwide_gross"] / out["production_budget"]   # = roi + 1
    out["selected"] = out["production_budget"] <= max_budget
    if genre != ALL_GENRES:
        out["selected"] &= out[genre] == 1
    out["breakout"] = out["multiple"] >= multiple
    return out


def breakout_summary(tagged: pd.DataFrame) -> dict:
    """Headline numbers for the selected films (expects the output of breakout_tag)."""
    sel, rest = tagged[tagged["selected"]], tagged[~tagged["selected"]]
    return {
        "films": len(sel),
        "breakouts": int(sel["breakout"].sum()),
        "rate": sel["breakout"].mean() if len(sel) else np.nan,
        "rest_rate": rest["breakout"].mean() if len(rest) else np.nan,
        "median_multiple": sel["multiple"].median() if len(sel) else np.nan,
        "lost_money": (sel["multiple"] < 1).mean() if len(sel) else np.nan,
    }


def breakout_figure(df: pd.DataFrame, max_budget: float = 9e6, genre: str = ALL_GENRES,
                    multiple: float = 6) -> go.Figure:
    """Budget vs. worldwide gross, with the chosen budget/genre slice highlighted."""
    tagged = breakout_tag(df, max_budget, genre, multiple)
    plot = tagged[tagged["worldwide_gross"] >= 1000]   # keep the log axis readable
    blue, orange = PALETTE[0], PALETTE[1]
    groups = [
        ("All other films", plot[~plot["selected"]], REF_COLOR, 5, 0.25),
        ("In your selection", plot[plot["selected"] & ~plot["breakout"]], blue, 7, 0.6),
        (f"Breakouts ({multiple}x+ budget)", plot[plot["selected"] & plot["breakout"]], orange, 9, 0.9),
    ]

    fig = go.Figure()
    for name, g, color, size, opacity in groups:
        fig.add_trace(go.Scatter(
            x=g["production_budget"], y=g["worldwide_gross"], mode="markers",
            name=f"{name} (n={len(g)})",
            marker=dict(color=color, size=size, opacity=opacity),
            customdata=np.column_stack([g["movie"], g["year"], g["genres"], g["multiple"]]),
            hovertemplate=("<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[2]}<br>"
                           "Budget %{x:$,.0f}<br>Gross %{y:$,.0f}<br>"
                           "<b>%{customdata[3]:.1f}x</b> budget<extra></extra>"),
        ))

    # Reference lines: break-even, the breakout line, and the budget cap
    x = np.array([2e4, 5e8])
    lines = [(x, x, REF_COLOR, "dash"), (x, x * multiple, orange, "dash"),
             ([max_budget, max_budget], [1e3, 4e9], REF_COLOR, "dot")]
    for lx, ly, color, dash in lines:
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=color, width=1, dash=dash),
                                 hoverinfo="skip", showlegend=False))
    labels = [(5e8, 5e8 * 0.5, "break-even", "right"), (2e4, 2e4 * multiple * 2, f"{multiple}x budget", "left"),
              (max_budget, 4e9, f" budget cap {fmt_money(max_budget)}", "left")]
    for lx, ly, text, anchor in labels:
        fig.add_annotation(x=np.log10(lx), y=np.log10(ly), text=text, showarrow=False,
                           xanchor=anchor, font=dict(size=11, color=REF_COLOR))

    axis = dict(type="log", tickvals=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
                ticktext=["$10K", "$100K", "$1M", "$10M", "$100M", "$1B"])
    fig.update_layout(
        xaxis=dict(title="Production budget (log scale)", range=[4.2, 8.8], **axis),
        yaxis=dict(title="Worldwide gross (log scale)", range=[3, 9.7], **axis),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=560, margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(align="left"),
    )
    return fig

# ------------------------------------------------------------
# ROI Relationship Analysis
# ------------------------------------------------------------
ROI_VARIABLES = {
    "Popularity": "popularity",
    "Production Budget": "production_budget",
    "Domestic Gross": "domestic_gross",
    "Foreign Gross": "foreign_gross",
    "Worldwide Gross": "worldwide_gross",
    "Vote Average": "vote_average",
}


def prepare_roi_data(df):
    """
    Prepare one consistent dataset for ROI analysis.
    """

    variables = list(ROI_VARIABLES.values())

    data = df.copy()

    if "no_reported_box_office" in data.columns:
        data = data[
            data["no_reported_box_office"] == False
        ].copy()

    data = (
        data[variables + ["roi"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    return data


def roi_relationship_data(df, variable):
    """
    Calculate correlation and quartile ROI statistics.
    """

    data = prepare_roi_data(df)

    correlation = data[variable].corr(data["roi"])

    data["Level"] = pd.qcut(
        data[variable],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    summary = (
        data.groupby("Level", observed=True)
        .agg(
            min_value=(variable, "min"),
            max_value=(variable, "max"),
            median_roi=("roi", "median"),
            mean_roi=("roi", "mean"),
            movies=("roi", "size"),
        )
        .reset_index()
    )

    movie_count = len(data)

    return summary, correlation, movie_count


def format_range_value(value, display_name):
    """
    Format the quartile range values.
    """

    money_variables = {
        "Production Budget",
        "Domestic Gross",
        "Foreign Gross",
        "Worldwide Gross",
    }

    if display_name in money_variables:

        if abs(value) >= 1_000_000_000:
            return f"${value / 1_000_000_000:.2f}B"

        elif abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"

        elif abs(value) >= 1_000:
            return f"${value / 1_000:.1f}K"

        else:
            return f"${value:,.0f}"

    return f"{value:.2f}"


def create_range_table(summary, display_name):
    """
    Create the Low, Medium, High, Very High range table.
    """

    table = summary.copy()

    table["Range"] = table.apply(
        lambda row: (
            f"{format_range_value(row['min_value'], display_name)}"
            f" – "
            f"{format_range_value(row['max_value'], display_name)}"
        ),
        axis=1,
    )

    table["median_roi"] = table["median_roi"].round(2)
    table["mean_roi"] = table["mean_roi"].round(2)

    table = table[
        [
            "Level",
            "Range",
            "median_roi",
            "mean_roi",
            "movies",
        ]
    ]

    table = table.rename(
        columns={
            "median_roi": "Median ROI",
            "mean_roi": "Mean ROI",
            "movies": "Movies",
        }
    )

    return table


def correlation_description(correlation):
    """
    Describe correlation strength and direction.
    """

    abs_correlation = abs(correlation)

    if abs_correlation < 0.10:
        strength = "very weak"

    elif abs_correlation < 0.30:
        strength = "weak"

    elif abs_correlation < 0.50:
        strength = "moderate"

    elif abs_correlation < 0.70:
        strength = "strong"

    else:
        strength = "very strong"

    if correlation > 0:
        direction = "positive"

    elif correlation < 0:
        direction = "negative"

    else:
        direction = "neutral"

    return strength, direction


def roi_key_insight(
    summary,
    correlation,
    display_name,
    roi_statistic,
):
    """
    Generate a short automatic interpretation.
    """

    if roi_statistic == "Median ROI":
        value_column = "median_roi"
    else:
        value_column = "mean_roi"

    low_roi = summary.iloc[0][value_column]
    high_roi = summary.iloc[-1][value_column]

    if high_roi > low_roi:
        comparison = "higher"

    elif high_roi < low_roi:
        comparison = "lower"

    else:
        comparison = "approximately the same"

    strength, direction = correlation_description(
        correlation
    )

    insight = (
        f"The Very High {display_name.lower()} group has a "
        f"{comparison} {roi_statistic.lower()} than the Low group "
        f"({high_roi:.2f} compared with {low_roi:.2f}). "
        f"The Pearson correlation is {correlation:.3f}, "
        f"indicating a {strength} {direction} linear relationship."
    )

    return insight


def roi_comparison_metrics(summary, roi_statistic):
    """
    Find highest ROI group and calculate change from Low to Very High.
    """

    if roi_statistic == "Median ROI":
        value_column = "median_roi"
    else:
        value_column = "mean_roi"

    best_row = summary.loc[
        summary[value_column].idxmax()
    ]

    low_value = summary.iloc[0][value_column]
    very_high_value = summary.iloc[-1][value_column]

    roi_change = very_high_value - low_value

    return {
        "highest_group": str(best_row["Level"]),
        "highest_roi": float(best_row[value_column]),
        "roi_change": float(roi_change),
    }


def strongest_roi_relationship(df):
    """
    Find the characteristic with the strongest absolute correlation with ROI.
    """

    data = prepare_roi_data(df)

    correlations = {
        display_name: data[column].corr(data["roi"])
        for display_name, column in ROI_VARIABLES.items()
    }

    valid_correlations = {
        name: value
        for name, value in correlations.items()
        if pd.notna(value)
    }

    strongest_name = max(
        valid_correlations,
        key=lambda name: abs(valid_correlations[name]),
    )

    strongest_correlation = valid_correlations[
        strongest_name
    ]

    return strongest_name, float(strongest_correlation)


def roi_relationship_figure(
    df,
    variable,
    display_name,
    roi_statistic,
):
    """
    Create the interactive ROI relationship graph.
    """

    summary, correlation, movie_count = (
        roi_relationship_data(
            df,
            variable,
        )
    )

    if roi_statistic == "Median ROI":
        value_column = "median_roi"
    else:
        value_column = "mean_roi"

    plot_summary = summary.copy()

    plot_summary["range_label"] = plot_summary.apply(
        lambda row: (
            f"{format_range_value(row['min_value'], display_name)}"
            f" – "
            f"{format_range_value(row['max_value'], display_name)}"
        ),
        axis=1,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_summary["Level"],
            y=plot_summary[value_column],
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(width=3),
            customdata=plot_summary[
                [
                    "range_label",
                    "median_roi",
                    "mean_roi",
                    "movies",
                ]
            ],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Range: %{customdata[0]}<br>"
                "Median ROI: %{customdata[1]:.2f}<br>"
                "Mean ROI: %{customdata[2]:.2f}<br>"
                "Movies: %{customdata[3]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=(
            f"{display_name} vs ROI | "
            f"Pearson r = {correlation:.3f}"
        ),
        xaxis_title=f"{display_name} Level",
        yaxis_title=roi_statistic,
        template="plotly_white",
        height=550,
        hovermode="closest",
    )

    return fig, correlation, summary, movie_count

# ---------------------------------------------------------------- Q2: release-month seasonality
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_MIN_FILMS = 5   # months with fewer films than this are grayed out as unreliable
SEASON_MEASURES = {
    "Typical return": "median_multiple",   # median worldwide gross / budget
    "Hit rate": "hit_rate",                # share of films that grossed more than their budget
}


def season_films(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """Films with reported box office in the chosen genre, plus `multiple` (gross / budget)."""
    out = df[~df["no_reported_box_office"]].copy()   # a $0 gross would count as a flop
    if genre != ALL_GENRES:
        out = out[out[genre] == 1]
    out["multiple"] = out["worldwide_gross"] / out["production_budget"]
    return out


def season_table(films: pd.DataFrame) -> pd.DataFrame:
    """One row per release month: films released, typical return, hit rate, and the top film."""
    rows = []
    for m, name in enumerate(MONTHS, start=1):
        sub = films[films["month"] == m]
        top = sub.loc[sub["multiple"].idxmax()] if len(sub) else None
        rows.append({
            "month": name, "films": len(sub),
            "median_multiple": sub["multiple"].median() if len(sub) else np.nan,
            "hit_rate": (sub["multiple"] >= 1).mean() if len(sub) else np.nan,
            "top_film": f"{top['movie']} ({top['multiple']:.1f}x)" if top is not None else "",
        })
    table = pd.DataFrame(rows)
    table["reliable"] = table["films"] >= SEASON_MIN_FILMS
    return table


def season_figure(table: pd.DataFrame, measure: str, baseline: float, genre: str) -> go.Figure:
    """Bars per release month for the chosen measure, compared with the genre's year-round value."""
    col = SEASON_MEASURES[measure]
    is_rate = col == "hit_rate"
    blue, orange = PALETTE[0], PALETTE[1]
    colors = [REF_COLOR if not ok else orange if v > baseline else blue
              for v, ok in zip(table[col], table["reliable"])]
    values = [f"{v:.0%}" if is_rate else f"{v:.1f}x" for v in table[col].fillna(0)]
    notes = ["" if ok else "<br><i>Too few films to trust</i>" for ok in table["reliable"]]

    fig = go.Figure(go.Bar(
        x=table["month"], y=table[col], marker_color=colors, marker_line_width=0,
        customdata=np.column_stack([values, table["films"], table["top_film"], notes]),
        hovertemplate=(f"<b>%{{x}}</b><br>{measure}: <b>%{{customdata[0]}}</b><br>"
                       "%{customdata[1]} films released<br>Top film: %{customdata[2]}"
                       "%{customdata[3]}<extra></extra>"),
    ))
    label = "all films" if genre == ALL_GENRES else genre
    base_text = f"{baseline:.0%}" if is_rate else f"{baseline:.1f}x"
    fig.add_hline(y=baseline, line=dict(color=REF_COLOR, width=1.5, dash="dash"))
    fig.add_annotation(text=f"Dashed line = {label}, all year: {base_text}", showarrow=False,
                       xref="paper", yref="paper", x=0, y=1.06, xanchor="left",
                       font=dict(size=12, color=REF_COLOR))
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=40, b=10), bargap=0.25, showlegend=False,
        yaxis=dict(title="Share that made money" if is_rate else "Median gross / budget",
                   tickformat=".0%" if is_rate else None, ticksuffix="" if is_rate else "x",
                   rangemode="tozero"),
        hoverlabel=dict(align="left"),
    )
    return fig


# ---------------------------------------------------------------- Q3: number of genres
GENRE_COUNT_ORDER = ["1", "2", "3", "4", "5+"]
GENRE_COUNT_MIN_FILMS = 5   # bars with fewer films than this are grayed out as unreliable
GENRE_COUNT_MEASURES = {
    "Typical return": "median_multiple",        # median worldwide gross / budget
    "Median budget": "median_budget",
    "Median worldwide gross": "median_gross",
}


def genre_count_films(df: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Films with reported box office, plus `primary_genre` (first-listed genre),
    `genre_count` ("1" to "5+") and `multiple` (gross / budget). Optionally one primary genre only."""
    out = df[~df["no_reported_box_office"]].copy()   # a $0 gross would count as a flop
    genre_lists = out["genres"].str.split(",")
    out["primary_genre"] = genre_lists.str[0].str.strip()
    out["genre_count"] = genre_lists.str.len().clip(upper=5).map(lambda n: "5+" if n == 5 else str(n))
    out["multiple"] = out["worldwide_gross"] / out["production_budget"]
    if primary != ALL_GENRES:
        out = out[out["primary_genre"] == primary]
    return out


def genre_count_table(films: pd.DataFrame) -> pd.DataFrame:
    """One row per genre count: films, median budget, median gross, typical return."""
    table = (films.groupby("genre_count")
             .agg(films=("movie", "count"), median_budget=("production_budget", "median"),
                  median_gross=("worldwide_gross", "median"), median_multiple=("multiple", "median"))
             .reindex(GENRE_COUNT_ORDER).reset_index())
    table["films"] = table["films"].fillna(0).astype(int)
    table["reliable"] = table["films"] >= GENRE_COUNT_MIN_FILMS
    return table


def genre_count_figure(table: pd.DataFrame, measure: str, primary: str) -> go.Figure:
    """Bars by number of genres for the chosen measure (gray = too few films to trust)."""
    col = GENRE_COUNT_MEASURES[measure]
    is_money = col != "median_multiple"
    colors = [PALETTE[0] if ok else REF_COLOR for ok in table["reliable"]]
    notes = ["" if ok else "<br><i>Too few films to trust</i>" for ok in table["reliable"]]
    budgets = [fmt_money(v) if pd.notna(v) else "–" for v in table["median_budget"]]
    grosses = [fmt_money(v) if pd.notna(v) else "–" for v in table["median_gross"]]
    returns = [f"{v:.1f}x" if pd.notna(v) else "–" for v in table["median_multiple"]]

    fig = go.Figure(go.Bar(
        x=table["genre_count"], y=table[col], marker_color=colors, marker_line_width=0,
        customdata=np.column_stack([table["films"], budgets, grosses, returns, notes]),
        hovertemplate=("<b>%{x} genre(s)</b><br>%{customdata[0]} films<br>"
                       "Median budget: %{customdata[1]}<br>Median gross: %{customdata[2]}<br>"
                       "Typical return: %{customdata[3]}%{customdata[4]}<extra></extra>"),
    ))
    label = "all films" if primary == ALL_GENRES else f"primary genre: {primary}"
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=40, b=10), bargap=0.3, showlegend=False,
        title=dict(text=f"{measure} by number of genres ({label})", font=dict(size=14)),
        xaxis=dict(title="Number of genres the film is tagged with", type="category"),
        yaxis=dict(title=measure, rangemode="tozero",
                   tickprefix="$" if is_money else "", tickformat="~s" if is_money else None,
                   ticksuffix="" if is_money else "x"),
        hoverlabel=dict(align="left"),
    )
    return fig


# ---------------------------------------------------------------- outcome predictor
# Predicts the odds that a film idea busts, lands in the middle, or breaks out, using only what a
# studio knows BEFORE release. Two simple models are averaged: a logistic regression (steady,
# easy to explain) and a random forest (catches combinations the regression can't).
# Tested on films it never saw (rolling forward through 2014-2018), this beats the earlier
# budget + genre model: bust AUC 0.73 vs 0.69, and better-calibrated probabilities.
OUTCOME_LABELS = ["Bust", "Middle", "Breakout"]
OUTCOME_COLORS = ["#c4501b", "#d49a14", "#0b86a6"]   # burnt orange, mustard gold, peacock teal
OUTCOME_BREAKOUT_ROI = 5.0                           # breakout = grossed 6x its budget or more
OUTCOME_RATINGS = ["G", "PG", "PG-13", "R"]
OUTCOME_SOURCES = {                                  # label shown in the app -> column in the TMDB file
    "Book": "based_on_book", "Comic": "based_on_comic", "True story": "based_on_true_story",
    "Video game": "based_on_video_game", "Remake / reboot": "is_remake",
}
OUTCOME_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                  "September", "October", "November", "December"]
OUTCOME_BASE_FEATURES = ["log_budget"] + GENRES      # the earlier, simpler model
OUTCOME_FEATURES = (OUTCOME_BASE_FEATURES + ["holiday", "summer", "fall", "runtime"]
                    + [f"rated_{r}" for r in OUTCOME_RATINGS] + ["is_sequel"]
                    + list(OUTCOME_SOURCES.values()))


def load_outcome_films(source) -> pd.DataFrame:
    """2010-2018 films from the TMDB-enriched file that have a budget and a usable ROI."""
    films = pd.read_csv(source)
    films = films[films["year"].between(YEAR_MIN, YEAR_MAX) & (films["production_budget"] > 0)]
    films = films.dropna(subset=["roi"]).copy()   # drops films with no reported box office
    films[GENRES] = films[GENRES].fillna(0).astype(int)
    return films


def outcome_features(films: pd.DataFrame, runtime_fill: float) -> pd.DataFrame:
    """Model inputs, all known before release. Month becomes three season flags
    (holiday = Nov-Dec, summer = May-Jul, fall = Aug-Oct; winter/spring is the baseline)."""
    X = pd.DataFrame(index=films.index)
    X["log_budget"] = np.log10(films["production_budget"])
    for g in GENRES:
        X[g] = films[g]
    X["holiday"] = films["month"].isin([11, 12]).astype(int)
    X["summer"] = films["month"].isin([5, 6, 7]).astype(int)
    X["fall"] = films["month"].isin([8, 9, 10]).astype(int)
    X["runtime"] = films["runtime"].fillna(runtime_fill)
    for r in OUTCOME_RATINGS:
        X[f"rated_{r}"] = (films["us_certification"] == r).astype(int)
    X["is_sequel"] = films["is_sequel"].eq(True).astype(int)
    for col in OUTCOME_SOURCES.values():
        X[col] = films[col].eq(True).astype(int)
    return X[OUTCOME_FEATURES]


def outcome_classes(roi: pd.Series) -> np.ndarray:
    """0 = bust (ROI < 0), 1 = middle (0 <= ROI < 5), 2 = breakout (ROI >= 5)."""
    return np.select([roi < 0, roi < OUTCOME_BREAKOUT_ROI], [0, 1], default=2)


class OutcomeModel:
    """Average of a logistic regression and a random forest, predicting [bust, middle, breakout].
    The regression is scaled and shrunk (C=0.03) so ~35 inputs on ~1,450 films don't overfit;
    the forest uses leaves of at least 8 films for the same reason."""

    def __init__(self, features: list[str] = OUTCOME_FEATURES, use_forest: bool = True):
        self.features, self.use_forest = features, use_forest

    def fit(self, films: pd.DataFrame):
        self.runtime_fill_ = float(films["runtime"].median())
        X = outcome_features(films, self.runtime_fill_)[self.features]
        y = outcome_classes(films["roi"])
        self.logit_ = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000)).fit(X, y)
        self.forest_ = (RandomForestClassifier(n_estimators=300, min_samples_leaf=8, random_state=0, n_jobs=-1)
                        .fit(X, y) if self.use_forest else None)
        return self

    def predict_proba(self, films: pd.DataFrame) -> np.ndarray:
        X = outcome_features(films, self.runtime_fill_)[self.features]
        p = self.logit_.predict_proba(X)
        return (p + self.forest_.predict_proba(X)) / 2 if self.forest_ is not None else p


def train_outcome_model(films: pd.DataFrame) -> OutcomeModel:
    return OutcomeModel().fit(films)


def predict_outcomes(model: OutcomeModel, budget_millions: float, genres: list[str], month: int,
                     rating: str, runtime: float, is_sequel: bool, sources: list[str]) -> np.ndarray:
    """Chances of [bust, middle, breakout] for one film idea; they sum to 1."""
    row = {"production_budget": budget_millions * 1e6, "month": month, "runtime": runtime,
           "us_certification": rating, "is_sequel": is_sequel}
    row.update({g: int(g in genres) for g in GENRES})
    row.update({col: label in sources for label, col in OUTCOME_SOURCES.items()})
    return model.predict_proba(pd.DataFrame([row]))[0]


def outcome_base_rates(films: pd.DataFrame) -> np.ndarray:
    """Share of all films that busted / landed in the middle / broke out (for reference)."""
    classes = outcome_classes(films["roi"])
    return np.bincount(classes, minlength=3) / len(classes)


def evaluate_outcome_model(films: pd.DataFrame, first_test_year: int = 2014) -> pd.DataFrame:
    """Rolling-forward test: for each year from first_test_year, train only on earlier films and
    predict that year. Compares a flat guess, the earlier budget + genre regression, and this model.
    Log loss: lower is better. AUC: 0.5 = coin flip, 1.0 = perfect ranking."""
    from sklearn.metrics import log_loss, roc_auc_score

    contenders = {
        "Flat guess (same odds for every film)": None,
        "Budget + genre only (earlier model)": lambda: OutcomeModel(OUTCOME_BASE_FEATURES, use_forest=False),
        "Current model (adds season, runtime, rating, sequel, source)": lambda: OutcomeModel(),
    }
    ys, preds = [], {name: [] for name in contenders}
    for year in range(first_test_year, YEAR_MAX + 1):
        train, test = films[films["year"] < year], films[films["year"] == year]
        y_train = outcome_classes(train["roi"])
        ys.append(outcome_classes(test["roi"]))
        for name, make in contenders.items():
            preds[name].append(np.tile(np.bincount(y_train, minlength=3) / len(y_train), (len(test), 1))
                               if make is None else make().fit(train).predict_proba(test))
    y = np.concatenate(ys)
    rows = []
    for name, p in preds.items():
        p, flat = np.vstack(p), contenders[name] is None   # a flat guess ranks nothing: AUC is 0.5 by definition
        rows.append({"Model": name, "Log loss": log_loss(y, p, labels=[0, 1, 2]),
                     "Bust AUC": 0.5 if flat else roc_auc_score(y == 0, p[:, 0]),
                     "Breakout AUC": 0.5 if flat else roc_auc_score(y == 2, p[:, 2])})
    return pd.DataFrame(rows)


def outcome_figure(probs: np.ndarray) -> go.Figure:
    """Bar chart of the three outcome chances, fixed order, one color per outcome."""
    fig = go.Figure(go.Bar(
        x=OUTCOME_LABELS, y=probs, marker_color=OUTCOME_COLORS,
        text=[f"{v:.0%}" for v in probs], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(range=[0, 1], tickformat=".0%", title="Chance"), xaxis=dict(title=None),
        margin=dict(t=20, b=10), height=380, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
    return fig


# ---------------------------------------------------------------- Q2: studio budget planner (TH)
# A studio splits its annual budget across film TYPES (strategies such as "low-budget horror"),
# not individual films. Each type's range of outcomes comes from the real 2010-2018 films that fit
# it: a simulated year draws each planned film's gross / budget from those films at random.
# Running the year thousands of times shows the slate's typical year, bad year and good year.
# Films are treated as independent, so a slate looks safer than it would in a bad industry year.
PLANNER_TYPES = {   # name -> (what it means, which films count). Budgets in millions.
    "Low-budget horror": ("Horror, up to $15M", lambda f, b: (f["Horror"] == 1) & (b <= 15)),
    "Indie drama": ("Drama (not horror), up to $15M",
                    lambda f, b: (f["Drama"] == 1) & (f["Horror"] == 0) & (b <= 15)),
    "Low-budget comedy": ("Live-action comedy, up to $25M",
                          lambda f, b: (f["Comedy"] == 1) & (f["Animation"] == 0) & (b <= 25)),
    "Mid-budget drama": ("Drama, $15-60M", lambda f, b: (f["Drama"] == 1) & b.between(15, 60)),
    "Mid-budget thriller": ("Thriller or crime (not horror), $15-60M",
                            lambda f, b: (f["Thriller"].eq(1) | f["Crime"].eq(1)) & (f["Horror"] == 0)
                            & b.between(15, 60)),
    "Mid-budget comedy": ("Live-action comedy, $25-80M",
                          lambda f, b: (f["Comedy"] == 1) & (f["Animation"] == 0) & b.between(25, 80)),
    "Mid-budget action": ("Action, $30-100M", lambda f, b: (f["Action"] == 1) & b.between(30, 100)),
    "Animated family film": ("Animation, $50M or more", lambda f, b: (f["Animation"] == 1) & (b >= 50)),
    "Original blockbuster": ("Not a sequel or franchise film, $100M or more",
                             lambda f, b: ~f["franchise_film"] & (b >= 100)),
    "Franchise blockbuster": ("Sequel or franchise film, $100M or more",
                              lambda f, b: f["franchise_film"] & (b >= 100)),
}
PLANNER_RISK_LEVELS = {   # risk level -> (what it optimizes, plain-English goal)
    "Conservative": ("safe", "the **lowest chance of losing money**"),
    "Balanced": ("typical_year", "the best **typical year** (the median year)"),
    "Aggressive": ("good_year", "the best **good year** (the best 1 in 10 years)"),
}
PLANNER_MAX_PER_TYPE = 10   # most films of one type the app can simulate
PLANNER_BREAKEVEN = {   # label -> gross needed per $1 of production budget to break even
    "1x budget (our bust line)": 1.0,
    "2.5x budget (adds marketing and the theaters' cut)": 2.5,
}
PLANNER_SIMS = 4000


def planner_types(films: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """One row per film type: its typical cost and how its real films did, plus each type's
    gross / budget multiples (the outcomes a simulated year draws from).
    A film can fit more than one type, just as it counts toward several genres."""
    f = films.copy()
    f["franchise_film"] = f["is_sequel"].eq(True) | f["franchise"].notna()
    millions = f["production_budget"] / 1e6
    multiple = f["worldwide_gross"] / f["production_budget"]
    rows, multiples = [], {}
    for name, (meaning, rule) in PLANNER_TYPES.items():
        fit = rule(f, millions)
        m = multiple[fit].to_numpy(float)
        multiples[name] = m
        rows.append({"Film type": name, "What counts": meaning, "Films": int(fit.sum()),
                     "Cost": float(f.loc[fit, "production_budget"].median()),
                     "Typical return": float(np.median(m)), "Bust": float((m < 1).mean()),
                     "Breakout": float((m >= OUTCOME_BREAKOUT_ROI + 1).mean())})
    return pd.DataFrame(rows), multiples


def planner_draws(multiples: dict[str, np.ndarray], max_per_type: int = PLANNER_MAX_PER_TYPE,
                  sims: int = PLANNER_SIMS, seed: int = 0) -> np.ndarray:
    """Simulated outcomes, shape (types, sims, max_per_type + 1): entry [t, s, k] is the total
    gross / budget of k films of type t in simulated year s (k = 0 gives 0). Every slate is scored
    on the same simulated years, so differences between slates aren't luck of the draw."""
    rng = np.random.default_rng(seed)
    out = np.zeros((len(multiples), sims, max_per_type + 1))
    for t, m in enumerate(multiples.values()):
        out[t, :, 1:] = rng.choice(m, size=(sims, max_per_type)).cumsum(axis=1)
    return out


def slate_returns(slates: np.ndarray, costs: np.ndarray, draws: np.ndarray, budget: float) -> np.ndarray:
    """Return on the annual budget for each slate in each simulated year, shape (sims, slates):
    (box office gross + any unspent money) / budget. Unspent money is simply kept, so leaving part
    of the budget unused isn't rewarded. A slate over budget is measured against what it spent.
    slates is (slates, types): how many films of each type."""
    slates = np.atleast_2d(slates)
    gross = sum(costs[t] * draws[t][:, slates[:, t]] for t in range(len(costs)))
    spent = slates @ costs
    return (gross + np.maximum(budget - spent, 0)) / np.maximum(budget, spent)


def slate_summary(returns: np.ndarray, breakeven: float = 1.0) -> dict:
    """Typical, bad and good year for one slate's simulated returns, and the chance of losing money."""
    return {"typical_year": float(np.median(returns)), "bad_year": float(np.quantile(returns, 0.1)),
            "good_year": float(np.quantile(returns, 0.9)), "lose_money": float((returns < breakeven).mean())}


def candidate_slates(costs: np.ndarray, budget: float, max_per_type: int, max_films: int,
                     n: int = 3000, seed: int = 1) -> np.ndarray:
    """Random slates that use up the budget: add a random affordable film type until nothing else fits
    (or the slate hits max_films). Returns the distinct slates, shape (slates, types)."""
    rng = np.random.default_rng(seed)
    slates = np.zeros((n, len(costs)), dtype=int)
    for i in range(n):
        left = budget
        while slates[i].sum() < max_films:
            fits = np.flatnonzero((costs <= left) & (slates[i] < max_per_type))
            if fits.size == 0:
                break
            t = rng.choice(fits)
            slates[i, t] += 1
            left -= costs[t]
    return np.unique(slates[slates.sum(axis=1) > 0], axis=0)


def recommend_slates(costs: np.ndarray, draws: np.ndarray, budget: float, max_per_type: int,
                     max_films: int, breakeven: float = 1.0) -> dict[str, np.ndarray] | None:
    """The best candidate slate for each risk level (see PLANNER_RISK_LEVELS). Ties go to the slate
    with the better typical year. None if no film type is affordable."""
    slates = candidate_slates(costs, budget, max_per_type, max_films)
    if len(slates) == 0:
        return None
    returns = slate_returns(slates, costs, draws, budget)
    typical = np.median(returns, axis=0)
    goals = {"safe": -(returns < breakeven).mean(axis=0), "typical_year": typical,
             "good_year": np.quantile(returns, 0.9, axis=0)}
    return {risk: slates[np.lexsort((typical, goals[key]))[-1]]
            for risk, (key, _) in PLANNER_RISK_LEVELS.items()}


def slate_figure(returns: np.ndarray, breakeven: float, compare: np.ndarray | None = None) -> go.Figure:
    """Histogram of one slate's simulated years (gross / money spent), with the break-even line.
    compare (optional) is drawn behind it as an outline, e.g. the recommended slate."""
    cap = float(np.quantile(returns if compare is None else np.concatenate([returns, compare]), 0.99))
    bins = dict(start=0, end=cap, size=cap / 60)
    fig = go.Figure()
    if compare is not None:
        fig.add_histogram(x=np.clip(compare, 0, cap), xbins=bins, name="Recommended slate",
                          marker=dict(color="rgba(0,0,0,0)", line=dict(color=REF_COLOR, width=1.5)),
                          histnorm="percent", hovertemplate="%{y:.1f}% of years<extra>Recommended</extra>")
    fig.add_histogram(x=np.clip(returns, 0, cap), xbins=bins, name="Your slate", histnorm="percent",
                      marker_color=OUTCOME_COLORS[2], opacity=0.85,
                      hovertemplate="Return %{x:.1f}x: %{y:.1f}% of years<extra></extra>")
    fig.add_vline(x=breakeven, line=dict(color=OUTCOME_COLORS[0], dash="dash", width=2),
                  annotation_text=f"Break even ({breakeven:g}x)", annotation_position="top right")
    fig.update_layout(
        barmode="overlay", height=380, margin=dict(t=30, b=10),
        xaxis=dict(title="Return on the annual budget in a simulated year", ticksuffix="x"),
        yaxis=dict(title="Share of simulated years", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        showlegend=compare is not None, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)")
    return fig
