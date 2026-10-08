"""Q1 interactive visual: the low-budget breakout finder (TH).

Question: which low-budget films made far more than they cost, and do they
have anything in common?

The data and chart logic live in apputil.py (the "Q1: low-budget breakouts"
section). This file is only the Streamlit page. Run it from this folder with:
    streamlit run q1_app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st

from apputil import *

st.set_page_config(page_title="Low-budget breakout finder", page_icon="🎬", layout="wide")
st.title("🎬 Low-budget breakout finder")
st.caption("Which low-budget films earned far more than they cost, and what do they have in common? "
           "Set a budget and a genre to see how often that kind of film breaks out.")

df = st.cache_data(load_data)(Path(__file__).parent / "movie_roi.csv")

# ---------------------------------------------------------------- controls
c1, c2, c3 = st.columns(3)
max_budget = c1.select_slider("Budget cap", BUDGET_STEPS, value=9e6, format_func=fmt_money,
                              help="Only films made for this much or less count as 'your selection'.")
genre = c2.selectbox("Genre", [ALL_GENRES] + GENRES,
                     help="A film counts toward every genre it is tagged with.")
multiple = c3.slider("Breakout multiple", 2, 20, 6, format="%dx",
                     help="A film is a breakout if its worldwide gross is at least this many times its budget. "
                          "6x is the same as ROI >= 5.")

tagged = breakout_tag(df, max_budget, genre, multiple)
s = breakout_summary(tagged)
label = "films" if genre == ALL_GENRES else f"{genre} films"
if s["films"] == 0:
    st.warning(f"No {label} in the data with a budget of {fmt_money(max_budget)} or less.")
    st.stop()

# ---------------------------------------------------------------- headline numbers
m1, m2, m3, m4 = st.columns(4)
m1.metric("Films in your selection", f"{s['films']:,}")
m2.metric("Breakout rate", f"{s['rate']:.0%}",
          delta=None if np.isnan(s["rest_rate"]) else f"{(s['rate'] - s['rest_rate']) * 100:+.0f} pts vs. all other films",
          help=f"{s['breakouts']} of {s['films']} grossed at least {multiple}x their budget.")
m3.metric("Typical return", f"{s['median_multiple']:.1f}x budget", help="Median worldwide gross / budget.")
m4.metric("Lost money", f"{s['lost_money']:.0%}", help="Share that grossed less than their budget.")
if s["films"] < 10:
    st.caption(f"Only {s['films']} films match, so these rates are not reliable.")

# ---------------------------------------------------------------- chart
st.plotly_chart(breakout_figure(df, max_budget, genre, multiple), width="stretch")
st.caption(f"{len(tagged):,} films released {YEAR_MIN}–{YEAR_MAX} with reported box office. Hover a dot to see the film.")

with st.expander(f"The {s['breakouts']} breakout films"):
    hits = tagged[tagged["selected"] & tagged["breakout"]].sort_values("multiple", ascending=False)
    st.dataframe(
        hits[["movie", "year", "genres", "production_budget", "worldwide_gross", "multiple"]],
        hide_index=True, width="stretch",
        column_config={
            "movie": "Film", "year": st.column_config.NumberColumn("Year", format="%d"), "genres": "Genres",
            "production_budget": st.column_config.NumberColumn("Budget", format="dollar"),
            "worldwide_gross": st.column_config.NumberColumn("Worldwide gross", format="dollar"),
            "multiple": st.column_config.NumberColumn("Gross / budget", format="%.1fx"),
        })
