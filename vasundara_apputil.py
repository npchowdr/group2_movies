import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_data(source):
    """Load the movie ROI dataset."""
    return pd.read_csv(source)


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

    # Keep the selected variable and ROI
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
        duplicates="drop"
    )

    # Calculate ROI statistics for each group
    summary = (
        data.groupby("Level", observed=True)
        .agg(
            median_roi=("roi", "median"),
            mean_roi=("roi", "mean"),
            movies=("roi", "size")
        )
        .reset_index()
    )

    return summary, correlation


def roi_relationship_figure(df, variable, display_name):
    """Create the interactive Plotly ROI visualization."""

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
            )
        )
    )

    fig.update_layout(
        title=f"{display_name} vs ROI | Pearson r = {correlation:.3f}",
        xaxis_title=f"{display_name} Level",
        yaxis_title="Median ROI",
        template="plotly_white",
        height=550,
        hovermode="closest"
    )

    return fig, correlation, summary