import streamlit as st

from apputil import *


from pathlib import Path

import streamlit as st

from apputil import (
    DEFAULT_GENRE, GENRES, YEAR_MAX, YEAR_MIN,
    extremes_figure, filter_data, fmt_money, genre_extremes, get_genre_risk_reward,
    load_data, risk_reward_figure, risk_reward_table,
)
st.set_page_config(page_title="Genre Risk vs. Reward", page_icon="🎲", layout="wide")
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
    exclude_no_bo = st.checkbox("Exclude films with no reported box office", value=False,
                                help="97 films in 2010–2018 have a budget but $0 gross.")
    min_films = st.slider("Minimum films per genre", 5, 30, 10,
                          help="Genres with fewer films give unreliable volatility estimates.")

    st.header("Spotlight genre")
    genre_choice = st.selectbox("Genre", GENRES, index=GENRES.index(DEFAULT_GENRE))
data = filter_data(df, years, exclude_no_bo)


# ---------------------------------------------------------------- header
st.title("🎲 Genre Risk vs. Reward")
st.caption(
    f"{len(data):,} films released {years[0]}–{years[1]}. Films with multiple genres "
    "count toward each of their genres."
)
risk = get_genre_risk_reward(data, GENRES, min_films)
if risk.empty:
    st.warning("No genres meet the minimum-films threshold for this year range.")
    st.stop()


# ---------------------------------------------------------------- tabs
tab_risk, tab_extremes, tab_stats = st.tabs(
    ["📊 Risk vs. reward", "⭐ Blockbuster vs. bust", "📋 Summary stats"]
)
with tab_risk:
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
with tab_stats:
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
    