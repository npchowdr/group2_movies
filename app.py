from pathlib import Path

import numpy as np
import streamlit as st

from apputil import *

st.set_page_config(page_title="Budget Distribution by Genre", page_icon="🎬", layout="wide")

cached_load = st.cache_data(load_data)

# ---------------------------------------------------------------- data source
default_path = Path(__file__).parent / "movie_roi.csv"
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("movie_roi.csv", type="csv",
                                help="Only needed if the CSV isn't next to this script.")
source = uploaded if uploaded is not None else (default_path if default_path.exists() else None)
if source is None:
    st.info("Upload **movie_roi.csv** in the sidebar to get started.")
    st.stop()
df = cached_load(source)

# ---------------------------------------------------------------- controls
with st.sidebar:
    st.header("Filters")
    years = st.slider("Release years", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
    genres = st.multiselect("Genres", GENRES, default=DEFAULT_GENRES)
    exclude_no_bo = st.checkbox("Exclude films with no reported box office", value=False,
                                help="97 films in 2010–2018 have a budget but $0 gross.")

    st.header("Display")
    log_scale = st.toggle("Log budget axis", value=True,
                          help="Budgets are heavily right-skewed; log scale shows the shape far better.")
    style = st.radio("Chart type", ["Density curve", "Histogram", "Both"], horizontal=True)
    height_mode = st.radio("Curve height", ["Normalized (compare shape)", "Scaled by # of films"],
                           help="Normalized: each curve has area 1. Scaled: taller curves = more films.")
    bw_adjust = st.slider("Curve smoothing", 0.3, 2.0, 1.0, 0.1)
    show_medians = st.checkbox("Show median lines", value=True)
    show_all = st.checkbox("Show 'All films' reference curve", value=True)

data = filter_data(df, years, exclude_no_bo)
all_budgets = data["production_budget"].to_numpy(float)

# ---------------------------------------------------------------- header
st.title("🎬 Production Budget Distribution by Genre")
st.caption(
    f"{len(data):,} films released {years[0]}–{years[1]}. Films with multiple genres "
    "count toward each of their genres, so genre curves overlap."
)

if not genres:
    st.warning("Pick at least one genre in the sidebar.")
    st.stop()

groups = genre_budgets(data, genres)
if not groups:
    st.warning("Not enough films for the selected genres and years.")
    st.stop()

# ---------------------------------------------------------------- tabs
tab_curve, tab_ridge, tab_stats = st.tabs(["📈 Distribution curves", "🎻 Side-by-side", "📋 Summary stats"])

with tab_curve:
    fig = distribution_figure(
        groups, all_budgets,
        log_scale=log_scale, style=style, scaled=height_mode.startswith("Scaled"),
        bw_adjust=bw_adjust, show_medians=show_medians, show_all=show_all,
    )
    st.plotly_chart(fig, width="stretch")
    if show_medians:
        st.caption("Dashed vertical lines mark each genre's median budget.")

with tab_ridge:
    st.plotly_chart(violin_figure(data, groups, log_scale), width="stretch")
    st.caption("Sorted by median budget. Hover over points to see individual films.")

with tab_stats:
    stats = summary_stats(groups)
    st.dataframe(
        stats_for_display(stats),
        hide_index=True,
        width="stretch",
        column_config={
            **{c: st.column_config.NumberColumn(c, format="$%.1fM") for c in MONEY_COLS},
            "% under $10M": st.column_config.NumberColumn(format="%.0f%%"),
            "% over $100M": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )

    top, bottom = stats.iloc[0], stats.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest median budget", top["Genre"], fmt_money(top["Median"]), delta_color="off")
    c2.metric("Lowest median budget", bottom["Genre"], fmt_money(bottom["Median"]), delta_color="off")
    c3.metric("All films median", fmt_money(float(np.median(all_budgets))))

    st.subheader("Average budget and standard deviation by genre")
    st.caption("All genres, using the year range and box-office filter from the sidebar.")
    st.dataframe(
        mean_std_table(data),
        hide_index=True,
        width="stretch",
        column_config={
            "Average budget": st.column_config.NumberColumn(format="$%.1fM"),
            "Std deviation": st.column_config.NumberColumn(format="$%.1fM"),
        },
    )