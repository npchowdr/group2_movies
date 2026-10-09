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


# ---------------------------------------------------------------- Q1: low-budget breakouts (TH)
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
