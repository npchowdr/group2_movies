"""
Helper functions for the movie budget distribution app.
Pure data / plotting logic — no Streamlit calls — so it can be imported and tested anywhere.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

YEAR_MIN, YEAR_MAX = 2010, 2018  # years with a meaningful sample (~120-200 films each)
GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]  # "TV Movie" omitted: only 3 films in range
DEFAULT_GENRES = ["Action", "Animation", "Comedy", "Drama", "Horror"]
PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#EECA3B", "#BAB0AC", "#1F77B4", "#D62728",
]
REF_COLOR = "#888888"


# ---------------------------------------------------------------- data
def load_data(source) -> pd.DataFrame:
    """Read the CSV and keep 2010–2018 films with a positive budget."""
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


# ------------------------------------------------------------
# ROI relationship visualization
# ------------------------------------------------------------

ROI_VARIABLES = {
    "Popularity": "popularity",
    "Production Budget": "production_budget",
    "Domestic Gross": "domestic_gross",
    "Foreign Gross": "foreign_gross",
    "Worldwide Gross": "worldwide_gross",
    "Vote Average": "vote_average",
}


def roi_relationship_data(df, variable):
    """Prepare quartile groups and ROI summary statistics."""

    data = df.copy()

    # Keep only movies with reported box-office information
    if "no_reported_box_office" in data.columns:
        data = data[data["no_reported_box_office"] == False]

    # Keep only the selected variable and ROI
    data = (
        data[[variable, "roi"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    # Pearson correlation using individual movies
    correlation = data[variable].corr(data["roi"])

    # Divide movies into four approximately equal groups
    data["Level"] = pd.qcut(
        data[variable],
        q=4,
        labels=["Low", "Medium", "High", "Very High"],
        duplicates="drop",
    )

    # Calculate summary values for each group
    summary = (
        data.groupby("Level", observed=True)
        .agg(
            median_roi=("roi", "median"),
            mean_roi=("roi", "mean"),
            movies=("roi", "size"),
        )
        .reset_index()
    )

    return summary, correlation


def roi_relationship_figure(df, variable, display_name):
    """Create an interactive Plotly chart for a movie variable vs ROI."""

    summary, correlation = roi_relationship_data(df, variable)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=summary["Level"],
            y=summary["median_roi"],
            mode="lines+markers",
            marker=dict(size=12),
            line=dict(width=3),
            customdata=summary[["mean_roi", "movies"]],
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Median ROI: %{y:.2f}<br>"
                "Mean ROI: %{customdata[0]:.2f}<br>"
                "Movies: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"{display_name} vs ROI | Pearson r = {correlation:.3f}",
        xaxis_title=f"{display_name} Level",
        yaxis_title="Median ROI",
        template="plotly_white",
        height=500,
        hovermode="closest",
    )

    return fig, correlation, summary
