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