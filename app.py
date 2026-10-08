from pathlib import Path

import numpy as np
import streamlit as st

from apputil import *

st.set_page_config(page_title="Movie Budgets & ROI by Genre", page_icon="🎬", layout="wide")

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
    # Shared by every tab
    st.header("Filters")
    years = st.slider("Release years", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
    exclude_no_bo = st.checkbox("Exclude films with no reported box office", value=False,
                                help="97 films in 2010–2018 have a budget but $0 gross.")

    # Budget distribution tabs
    with st.expander("🎬 Budget distribution options", expanded=True):
        genres = st.multiselect("Genres", GENRES, default=DEFAULT_GENRES)
        log_scale = st.toggle("Log budget axis", value=True,
                              help="Budgets are heavily right-skewed; log scale shows the shape far better.")
        style = st.radio("Chart type", ["Density curve", "Histogram", "Both"], horizontal=True)
        height_mode = st.radio("Curve height", ["Normalized (compare shape)", "Scaled by # of films"],
                               help="Normalized: each curve has area 1. Scaled: taller curves = more films.")
        bw_adjust = st.slider("Curve smoothing", 0.3, 2.0, 1.0, 0.1)
        show_medians = st.checkbox("Show median lines", value=True)
        show_all = st.checkbox("Show 'All films' reference curve", value=True)

    # Risk vs. reward tabs
    with st.expander("🎲 Risk vs. reward options", expanded=True):
        min_films = st.slider("Minimum films per genre", 5, 30, 10,
                              help="Genres with fewer films give unreliable volatility estimates.")
        genre_choice = st.selectbox("Spotlight genre", GENRES, index=GENRES.index(DEFAULT_GENRE))

data = filter_data(df, years, exclude_no_bo)
all_budgets = data["production_budget"].to_numpy(float)

# ---------------------------------------------------------------- header
st.title("🎬 Movie Budgets & ROI by Genre")
st.caption(
    f"{len(data):,} films released {years[0]}–{years[1]}. Films with multiple genres "
    "count toward each of their genres, so genre results overlap."
)

# Compute both datasets up front. Instead of st.stop(), problems are shown
# inside the affected tabs so the other section keeps working.
groups = genre_budgets(data, genres) if genres else None
if not genres:
    budget_problem = "Pick at least one genre in the sidebar."
elif not groups:
    budget_problem = "Not enough films for the selected genres and years."
else:
    budget_problem = None

risk = get_genre_risk_reward(data, GENRES, min_films)
risk_problem = ("No genres meet the minimum-films threshold for this year range."
                if risk.empty else None)

# ---------------------------------------------------------------- tabs
(tab_curve, tab_ridge, tab_budget_stats,
 tab_risk, tab_extremes, tab_risk_stats) = st.tabs([
    "📈 Budget curves", "🎻 Budget side-by-side", "📋 Budget stats",
    "📊 Risk vs. reward", "⭐ Blockbuster vs. bust", "📋 ROI stats",
])

# ======================= Budget distribution =======================
with tab_curve:
    if budget_problem:
        st.warning(budget_problem)
    else:
        fig = distribution_figure(
            groups, all_budgets,
            log_scale=log_scale, style=style, scaled=height_mode.startswith("Scaled"),
            bw_adjust=bw_adjust, show_medians=show_medians, show_all=show_all,
        )
        st.plotly_chart(fig, width="stretch")
        if show_medians:
            st.caption("Dashed vertical lines mark each genre's median budget.")

with tab_ridge:
    if budget_problem:
        st.warning(budget_problem)
    else:
        st.plotly_chart(violin_figure(data, groups, log_scale), width="stretch")
        st.caption("Sorted by median budget. Hover over points to see individual films.")

with tab_budget_stats:
    if budget_problem:
        st.warning(budget_problem)
    else:
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

    # This table covers all genres, so it doesn't depend on the genre selection
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

# ========================= Risk vs. reward =========================
with tab_risk:
    if risk_problem:
        st.warning(risk_problem)
    else:
        st.plotly_chart(risk_reward_figure(risk), width="stretch")
        st.caption(
            "Further right = higher average ROI. Lower = more consistent returns; "
            "higher = more boom-or-bust. Bubble size = number of films. Only standout "
            "genres are labeled on the chart — hover any bubble to see the rest."
        )

with tab_extremes:
    subset, biggest, worst = genre_extremes(data, genre_choice)
    st.plotly_chart(extremes_figure(subset, biggest, worst, genre_choice), width="stretch")
    st.caption(f"Every {genre_choice} film by budget and ROI, with the biggest-budget film "
               "and the worst-performing film called out.")
    c1, c2 = st.columns(2)
    c1.metric(f"Biggest {genre_choice} blockbuster", biggest["movie"],
              f"{fmt_money(biggest['production_budget'])} budget, ROI {biggest['roi']:.2f}",
              delta_color="off")
    c2.metric(f"Lowest {genre_choice} ROI", worst["movie"],
              f"{fmt_money(worst['production_budget'])} budget, ROI {worst['roi']:.2f}",
              delta_color="off")
    st.info(
        f"**{biggest['movie']}** ({int(biggest['year'])}) is {genre_choice}'s biggest bet: "
        f"{fmt_money(biggest['production_budget'])} of budget, returning {biggest['roi']:.1f}x. "
        f"**{worst['movie']}** ({int(worst['year'])}) is the cautionary tale: "
        f"{fmt_money(worst['production_budget'])} of budget, and it lost "
        f"{abs(worst['roi']) * 100:.0f}% of that. Same genre, opposite outcomes."
    )

with tab_risk_stats:
    if risk_problem:
        st.warning(risk_problem)
    else:
        st.dataframe(
            risk_reward_table(risk), hide_index=True, width="stretch",
            column_config={
                "Avg ROI": st.column_config.NumberColumn(format="%.2f"),
                "ROI Std Dev": st.column_config.NumberColumn(format="%.2f"),
                "Volatility (CV)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        most_consistent, most_volatile = risk.iloc[0], risk.iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Most consistent genre", most_consistent["genre"],
                  f"CV {most_consistent['coefficient_of_variation']:.2f}", delta_color="off")
        c2.metric("Most unpredictable genre", most_volatile["genre"],
                  f"CV {most_volatile['coefficient_of_variation']:.2f}", delta_color="off")


# from pathlib import Path

# import numpy as np
# import streamlit as st

# from apputil import *


# #st.set_page_config(page_title="Genre Risk vs. Reward", page_icon="🎲", layout="wide")
# st.set_page_config(page_title="Group 2", page_icon="🎲", layout="wide")
# cached_load = st.cache_data(load_data)



# #st.set_page_config(page_title="Budget Distribution by Genre", page_icon="🎬", layout="wide")

# cached_load = st.cache_data(load_data)

# # ---------------------------------------------------------------- data source
# default_path = Path(__file__).parent / "movie_roi.csv"
# with st.sidebar:
#     st.header("Data")
#     uploaded = st.file_uploader("movie_roi.csv", type="csv",
#                                 help="Only needed if the CSV isn't next to this script.")
# source = uploaded if uploaded is not None else (default_path if default_path.exists() else None)
# if source is None:
#     st.info("Upload **movie_roi.csv** in the sidebar to get started.")
#     st.stop()
# df = cached_load(source)


# # ---------------------------------------------------------------- controls
# with st.sidebar:
#     st.header("Filters")
#     years = st.slider("Release years", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
# <<<<<<< HEAD
#     exclude_no_bo = st.checkbox("Exclude films with no reported box office", value=False,
#                                 help="97 films in 2010–2018 have a budget but $0 gross.")
#     min_films = st.slider("Minimum films per genre", 5, 30, 10,
#                           help="Genres with fewer films give unreliable volatility estimates.")

#     st.header("Spotlight genre")
#     genre_choice = st.selectbox("Genre", GENRES, index=GENRES.index(DEFAULT_GENRE))
# data = filter_data(df, years, exclude_no_bo)


# # ---------------------------------------------------------------- header
# st.title("🎲 Genre Risk vs. Reward")
# st.caption(
#     f"{len(data):,} films released {years[0]}–{years[1]}. Films with multiple genres "
#     "count toward each of their genres."
# )
# risk = get_genre_risk_reward(data, GENRES, min_films)
# if risk.empty:
#     st.warning("No genres meet the minimum-films threshold for this year range.")
#     st.stop()


# # ---------------------------------------------------------------- tabs
# tab_risk, tab_extremes, tab_stats = st.tabs(
#     ["📊 Risk vs. reward", "⭐ Blockbuster vs. bust", "📋 Summary stats"]
# )
# with tab_risk:
#     st.plotly_chart(risk_reward_figure(risk), width="stretch")
#     st.caption(
#         "Further right = higher average ROI. Lower = more consistent returns; "
#         "higher = more boom-or-bust. Bubble size = number of films. Only standout "
#         "genres are labeled on the chart — hover any bubble to see the rest."
#     )
# with tab_extremes:
#     subset, biggest, worst = genre_extremes(data, genre_choice)
#     st.plotly_chart(extremes_figure(subset, biggest, worst, genre_choice), width="stretch")
#     st.caption(f"Every {genre_choice} film by budget and ROI, with the biggest-budget film "
#               "and the worst-performing film called out.")
#     c1, c2 = st.columns(2)
#     c1.metric(f"Biggest {genre_choice} blockbuster", biggest["movie"],
#              f"{fmt_money(biggest['production_budget'])} budget, ROI {biggest['roi']:.2f}",
#              delta_color="off")
#     c2.metric(f"Lowest {genre_choice} ROI", worst["movie"],
#              f"{fmt_money(worst['production_budget'])} budget, ROI {worst['roi']:.2f}",
#              delta_color="off")
#     st.info(
#         f"**{biggest['movie']}** ({int(biggest['year'])}) is {genre_choice}'s biggest bet: "
#         f"{fmt_money(biggest['production_budget'])} of budget, returning {biggest['roi']:.1f}x. "
#         f"**{worst['movie']}** ({int(worst['year'])}) is the cautionary tale: "
#         f"{fmt_money(worst['production_budget'])} of budget, and it lost "
#         f"{abs(worst['roi']) * 100:.0f}% of that. Same genre, opposite outcomes."
#     )
# with tab_stats:
#     st.dataframe(
#         risk_reward_table(risk), hide_index=True, width="stretch",
#         column_config={
#             "Avg ROI": st.column_config.NumberColumn(format="%.2f"),
#             "ROI Std Dev": st.column_config.NumberColumn(format="%.2f"),
#             "Volatility (CV)": st.column_config.NumberColumn(format="%.2f"),
#         },
#     )
#     most_consistent, most_volatile = risk.iloc[0], risk.iloc[-1]
#     c1, c2 = st.columns(2)
#     c1.metric("Most consistent genre", most_consistent["genre"],
#              f"CV {most_consistent['coefficient_of_variation']:.2f}", delta_color="off")
#     c2.metric("Most unpredictable genre", most_volatile["genre"],
#              f"CV {most_volatile['coefficient_of_variation']:.2f}", delta_color="off")
# =======
#     genres = st.multiselect("Genres", GENRES, default=DEFAULT_GENRES)
#     exclude_no_bo = st.checkbox("Exclude films with no reported box office", value=False,
#                                 help="97 films in 2010–2018 have a budget but $0 gross.")

#     st.header("Display")
#     log_scale = st.toggle("Log budget axis", value=True,
#                           help="Budgets are heavily right-skewed; log scale shows the shape far better.")
#     style = st.radio("Chart type", ["Density curve", "Histogram", "Both"], horizontal=True)
#     height_mode = st.radio("Curve height", ["Normalized (compare shape)", "Scaled by # of films"],
#                            help="Normalized: each curve has area 1. Scaled: taller curves = more films.")
#     bw_adjust = st.slider("Curve smoothing", 0.3, 2.0, 1.0, 0.1)
#     show_medians = st.checkbox("Show median lines", value=True)
#     show_all = st.checkbox("Show 'All films' reference curve", value=True)

# data = filter_data(df, years, exclude_no_bo)
# all_budgets = data["production_budget"].to_numpy(float)

# # ---------------------------------------------------------------- header
# st.title("🎬 Production Budget Distribution by Genre")
# st.caption(
#     f"{len(data):,} films released {years[0]}–{years[1]}. Films with multiple genres "
#     "count toward each of their genres, so genre curves overlap."
# )

# if not genres:
#     st.warning("Pick at least one genre in the sidebar.")
#     st.stop()

# groups = genre_budgets(data, genres)
# if not groups:
#     st.warning("Not enough films for the selected genres and years.")
#     st.stop()

# # ---------------------------------------------------------------- tabs
# tab_curve, tab_ridge, tab_stats = st.tabs(["📈 Distribution curves", "🎻 Side-by-side", "📋 Summary stats"])

# with tab_curve:
#     fig = distribution_figure(
#         groups, all_budgets,
#         log_scale=log_scale, style=style, scaled=height_mode.startswith("Scaled"),
#         bw_adjust=bw_adjust, show_medians=show_medians, show_all=show_all,
#     )
#     st.plotly_chart(fig, width="stretch")
#     if show_medians:
#         st.caption("Dashed vertical lines mark each genre's median budget.")

# with tab_ridge:
#     st.plotly_chart(violin_figure(data, groups, log_scale), width="stretch")
#     st.caption("Sorted by median budget. Hover over points to see individual films.")

# with tab_stats:
#     stats = summary_stats(groups)
#     st.dataframe(
#         stats_for_display(stats),
#         hide_index=True,
#         width="stretch",
#         column_config={
#             **{c: st.column_config.NumberColumn(c, format="$%.1fM") for c in MONEY_COLS},
#             "% under $10M": st.column_config.NumberColumn(format="%.0f%%"),
#             "% over $100M": st.column_config.NumberColumn(format="%.0f%%"),
#         },
#     )

#     top, bottom = stats.iloc[0], stats.iloc[-1]
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Highest median budget", top["Genre"], fmt_money(top["Median"]), delta_color="off")
#     c2.metric("Lowest median budget", bottom["Genre"], fmt_money(bottom["Median"]), delta_color="off")
#     c3.metric("All films median", fmt_money(float(np.median(all_budgets))))

#     st.subheader("Average budget and standard deviation by genre")
#     st.caption("All genres, using the year range and box-office filter from the sidebar.")
#     st.dataframe(
#         mean_std_table(data),
#         hide_index=True,
#         width="stretch",
#         column_config={
#             "Average budget": st.column_config.NumberColumn(format="$%.1fM"),
#             "Std deviation": st.column_config.NumberColumn(format="$%.1fM"),
#         },
#     )
# >>>>>>> b636cf76e1b603c7352b82ded1e4043404f382a6
    