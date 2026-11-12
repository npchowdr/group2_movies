from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
 
from apputil import *
st.set_page_config(page_title="Movie Budgets & ROI by Genre", page_icon="🎬", layout="wide")


# ---------------------------------------------------------------- background image
@st.cache_data
def _background_b64(path):
    import base64
    return base64.b64encode(Path(path).read_bytes()).decode()


_bg_path = Path(__file__).parent / "black_background_image.jpg"
# How much to dim the background image: 0 = full-strength image, 1 = solid black.
BG_DIM = 0.8
if _bg_path.exists():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(14, 17, 23, {BG_DIM}), rgba(14, 17, 23, {BG_DIM})),
                              url("data:image/jpeg;base64,{_background_b64(str(_bg_path))}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
 

cached_load = st.cache_data(load_data)
cached_outcome_films = st.cache_data(load_outcome_films)
cached_outcome_model = st.cache_resource(train_outcome_model)   # trained once, reused on every rerun
cached_planner_types = st.cache_data(planner_types)
cached_planner_draws = st.cache_data(planner_draws)
cached_recommend_slates = st.cache_data(recommend_slates)
 
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
# Fewer tabs, so there is less clicking: related views share a tab and read top to bottom.
# The old per-view names are kept as aliases, so each section below still says which tab it lives in.
tab_predictor, tab_planner, tab_budgets, tab_risk_reward, tab_timing = st.tabs([
    "🔮 Outcome predictor", "🗓️ Budget planner", "💰 Budgets", "⚖️ Risk & reward", "🚀 Breakouts & timing",
])
tab_curve = tab_budget_stats = tab_budgets
tab_risk = tab_extremes = tab_risk_stats = tab_roi = tab_risk_reward
tab_low_budget_breakouts = tab_release_month = tab_genre_count = tab_timing
 
# ===================== Outcome predictor =====================
with tab_predictor:
    st.subheader("Movie outcome predictor")
    st.markdown(
        "Describe a movie idea and see the odds that it **busts** (loses money), lands in the "
        "**middle**, or is a **breakout** (grosses 6x its budget or more)."
    )
    outcome_file = Path(__file__).parent / "data" / "movie_roi_tmdb.csv"
    if not outcome_file.exists():
        st.info("The predictor needs **data/movie_roi_tmdb.csv** (the TMDB-enriched file) next to this script.")
    else:
        # Trained on every 2010-2018 film, so the sidebar filters don't change the model.
        outcome_films = cached_outcome_films(outcome_file)
        outcome_model = cached_outcome_model(outcome_films)
 
        inputs, results = st.columns([1, 1.25], gap="large")
        with inputs:
            pred_budget = st.slider("Production budget (millions of dollars)", 1, 400, 60, key="pred_budget")
            pred_genres = st.multiselect("Genres", GENRES, default=["Action", "Adventure"], key="pred_genres")
            m1, m2 = st.columns(2)
            pred_month = m1.selectbox("Release month", OUTCOME_MONTHS, index=6, key="pred_month")
            pred_rating = m2.selectbox("Rating", OUTCOME_RATINGS, index=2, key="pred_rating")
            pred_runtime = st.slider("Runtime (minutes)", 70, 200, 115, key="pred_runtime")
            pred_sequel = st.toggle("Sequel or part of a franchise", value=False, key="pred_sequel")
            pred_sources = st.multiselect("Based on", list(OUTCOME_SOURCES), key="pred_sources",
                                          help="Leave empty for an original story.")
 
        pred_probs = predict_outcomes(
            outcome_model, pred_budget, pred_genres, OUTCOME_MONTHS.index(pred_month) + 1,
            pred_rating, pred_runtime, pred_sequel, pred_sources)
        with results:
            for col, label, prob in zip(st.columns(3), OUTCOME_LABELS, pred_probs):
                col.metric(label, f"{prob:.0%}")
            st.plotly_chart(outcome_figure(pred_probs), width="stretch")
 
        base = outcome_base_rates(outcome_films)
        st.caption(
            f"For reference, across all films from {YEAR_MIN}–{YEAR_MAX}: {base[0]:.0%} busted, "
            f"{base[1]:.0%} landed in the middle, {base[2]:.0%} were breakouts. The model only uses what "
            "a studio knows before release (budget, genre, release month, rating, runtime, sequel status "
            "and source material), so treat these as rough odds, not predictions."
        )
 
# ===================== Studio budget planner (TH) =====================
with tab_planner:
    st.subheader("Studio budget planner")
    st.markdown(
        "Split a year's production budget across **types of films** and see the range of years that "
        "slate could have. Pick a risk level for a recommended slate, then change the number of films "
        "of each type in the table to try your own."
    )
    planner_file = Path(__file__).parent / "data" / "movie_roi_tmdb.csv"
    if not planner_file.exists():
        st.info("The planner needs **data/movie_roi_tmdb.csv** (the TMDB-enriched file) next to this script.")
    else:
        # Uses every 2010-2018 film with reported box office, so the sidebar filters don't change it.
        film_types, type_multiples = cached_planner_types(cached_outcome_films(planner_file))
        type_costs = film_types["Cost"].to_numpy(float)
        type_draws = cached_planner_draws(type_multiples)

        p1, p2, p3 = st.columns([1.2, 1, 1.2])
        plan_budget = p1.slider("Annual production budget (millions of dollars)", 50, 2000, 500, 25,
                                key="plan_budget") * 1e6
        plan_risk = p2.radio("Risk level", list(PLANNER_RISK_LEVELS), index=1, key="plan_risk",
                             help="Conservative: lowest chance of losing money. Balanced: best typical "
                                  "year. Aggressive: best good year (the best 1 in 10).")
        breakeven_label = p3.radio("Break even at", list(PLANNER_BREAKEVEN), key="plan_breakeven",
                                   help="Production budget leaves out marketing (often as much again), and "
                                        "theaters keep about half the box office. 2.5x is a common rule of "
                                        "thumb for the gross a film needs to actually make money.")
        plan_breakeven = PLANNER_BREAKEVEN[breakeven_label]
        with st.expander("Slate limits"):
            l1, l2 = st.columns(2)
            plan_max_type = l1.slider("Most films of one type", 1, 5, 3, key="plan_max_type",
                                      help="Stops the planner from making ten horror films: audiences "
                                           "and release dates for one kind of film run out.")
            plan_max_films = l2.slider("Most films in the year", 4, 25, 15, key="plan_max_films",
                                       help="Large studios release roughly 10 to 25 films a year.")

        recs = cached_recommend_slates(type_costs, type_draws, plan_budget, plan_max_type,
                                       plan_max_films, plan_breakeven)
        if recs is None:
            st.warning(f"No film type costs {fmt_money(plan_budget)} or less.")
        else:
            def slate_text(slate):
                return ", ".join(f"{n} {name.lower()}" for name, n in zip(film_types["Film type"], slate) if n)

            rec_returns = {risk: slate_returns(s, type_costs, type_draws, plan_budget)[:, 0]
                           for risk, s in recs.items()}
            rec_stats = {risk: slate_summary(r, plan_breakeven) for risk, r in rec_returns.items()}

            st.markdown(f"**Recommended slate ({plan_risk}):** {slate_text(recs[plan_risk])}. "
                        f"Out of a few thousand possible slates, this one has "
                        f"{PLANNER_RISK_LEVELS[plan_risk][1]}.")

            # Editable slate: starts at the recommendation, and resets when any setting above changes.
            editor = film_types.assign(**{"Cost": film_types["Cost"] / 1e6, "Bust": film_types["Bust"] * 100,
                                          "Breakout": film_types["Breakout"] * 100, "Films planned": recs[plan_risk]})
            editor = editor[["Film type", "Films planned", "What counts", "Cost", "Typical return",
                             "Bust", "Breakout", "Films"]]
            edited = st.data_editor(
                editor, hide_index=True, width="stretch",
                key=f"plan_editor_{plan_budget}_{plan_risk}_{plan_max_type}_{plan_max_films}_{plan_breakeven}",
                disabled=[c for c in editor.columns if c != "Films planned"],
                column_config={
                    "Films planned": st.column_config.NumberColumn(
                        min_value=0, max_value=PLANNER_MAX_PER_TYPE, step=1, format="%d",
                        help="Change these to try your own slate."),
                    "Cost": st.column_config.NumberColumn("Typical cost", format="$%.1fM",
                                                          help="Median budget of real films of this type."),
                    "Typical return": st.column_config.NumberColumn(format="%.1fx",
                                                                    help="Median worldwide gross / budget."),
                    "Bust": st.column_config.NumberColumn("Bust rate", format="%.0f%%",
                                                          help="Share that grossed less than their budget."),
                    "Breakout": st.column_config.NumberColumn("Breakout rate", format="%.0f%%",
                                                              help="Share that grossed 6x their budget or more."),
                    "Films": st.column_config.NumberColumn("Real films (2010-18)", format="%d"),
                },
            )
            my_slate = edited["Films planned"].fillna(0).clip(0, PLANNER_MAX_PER_TYPE).astype(int).to_numpy()
            customized = not np.array_equal(my_slate, recs[plan_risk])

            if my_slate.sum() == 0:
                st.warning("Add at least one film to the slate.")
            else:
                my_returns = slate_returns(my_slate, type_costs, type_draws, plan_budget)[:, 0]
                mine, rec = slate_summary(my_returns, plan_breakeven), rec_stats[plan_risk]
                spent = float(my_slate @ type_costs)
                if spent > plan_budget:
                    st.warning(f"This slate costs {fmt_money(spent)}, which is "   # \$ so $ isn't read as math
                               f"{fmt_money(spent - plan_budget)} over budget.".replace("$", "\\$"))

                def vs_rec(key, pct=False):   # change from the recommended slate, shown only once edited
                    d = mine[key] - rec[key]
                    if not customized or abs(d) < 1e-9:
                        return None
                    return f"{d * 100:+.1f} pts vs. recommended" if pct else f"{d:+.2f}x vs. recommended"

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Spent", fmt_money(spent), f"{int(my_slate.sum())} films", delta_color="off")
                m2.metric("Typical year", f"{mine['typical_year']:.2f}x", vs_rec("typical_year"),
                          help="Median simulated year: (gross + unspent money) / budget.")
                m3.metric("Bad year", f"{mine['bad_year']:.2f}x", vs_rec("bad_year"),
                          help="1 in 10 simulated years is worse than this.")
                m4.metric("Good year", f"{mine['good_year']:.2f}x", vs_rec("good_year"),
                          help="1 in 10 simulated years is better than this.")
                m5.metric("Chance of losing money", f"{mine['lose_money']:.0%}", vs_rec("lose_money", pct=True),
                          delta_color="inverse",
                          help=f"Share of simulated years whose return is below {plan_breakeven:g}x.")

                st.plotly_chart(slate_figure(my_returns, plan_breakeven,
                                             rec_returns[plan_risk] if customized else None), width="stretch")

            st.markdown("**The three risk levels side by side**")
            st.dataframe(
                pd.DataFrame([{"Risk level": risk, "Slate": slate_text(s), "Films": int(s.sum()),
                               "Spent": float(s @ type_costs) / 1e6,
                               "Typical year": rec_stats[risk]["typical_year"],
                               "Bad year": rec_stats[risk]["bad_year"],
                               "Good year": rec_stats[risk]["good_year"],
                               "Chance of losing money": rec_stats[risk]["lose_money"] * 100}
                              for risk, s in recs.items()]),
                hide_index=True, width="stretch",
                column_config={
                    "Spent": st.column_config.NumberColumn(format="$%.0fM"),
                    **{c: st.column_config.NumberColumn(format="%.2fx")
                       for c in ["Typical year", "Bad year", "Good year"]},
                    "Chance of losing money": st.column_config.NumberColumn(format="%.0f%%"),
                },
            )
            st.caption(
                f"How it works: each film type's outcomes come from the real {YEAR_MIN}–{YEAR_MAX} films that "
                f"fit it (a film can fit more than one type). The app simulates the year {PLANNER_SIMS:,} "
                "times, each time drawing every planned film's gross / budget from those real films. "
                "Limits: films are treated as independent, so a big slate looks safer than it would in a bad "
                "year for the whole industry, and production budgets leave out marketing (try the 2.5x "
                "break-even to allow for it)."
            )

# ======================= Budget distribution =======================
with tab_curve:
    st.subheader("Budget distribution by genre")
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
 
with tab_budget_stats:
    st.divider()
    st.subheader("Budget stats")
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
    st.subheader("Risk vs. reward")
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
    st.divider()
    st.subheader("Blockbuster vs. bust")
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
    st.divider()
    st.subheader("ROI stats")
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
 
# ===================== Low-budget breakouts =====================
with tab_low_budget_breakouts:
    st.subheader("Low-budget breakout finder")
    st.caption("Which low-budget films earned far more than they cost, and what do they have in common? "
               "Set a budget and a genre to see how often that kind of film breaks out.")
 
    b1, b2, b3 = st.columns(3)
    max_budget = b1.select_slider("Budget cap", BUDGET_STEPS, value=9e6, format_func=fmt_money,
                                  help="Only films made for this much or less count as 'your selection'.")
    breakout_genre = b2.selectbox("Genre", [ALL_GENRES] + GENRES,
                                  help="A film counts toward every genre it is tagged with.")
    multiple = b3.slider("Breakout multiple", 2, 20, 6, format="%dx",
                         help="A film is a breakout if its worldwide gross is at least this many times "
                              "its budget. 6x is the same as ROI >= 5.")
 
    tagged = breakout_tag(data, max_budget, breakout_genre, multiple)
    s = breakout_summary(tagged)
    if s["films"] == 0:
        label = "films" if breakout_genre == ALL_GENRES else f"{breakout_genre} films"
        st.warning(f"No {label} in the data with a budget of {fmt_money(max_budget)} or less.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Films in your selection", f"{s['films']:,}")
        m2.metric("Breakout rate", f"{s['rate']:.0%}",
                  delta=None if np.isnan(s["rest_rate"])
                  else f"{(s['rate'] - s['rest_rate']) * 100:+.0f} pts vs. all other films",
                  help=f"{s['breakouts']} of {s['films']} grossed at least {multiple}x their budget.")
        m3.metric("Typical return", f"{s['median_multiple']:.1f}x budget",
                  help="Median worldwide gross / budget.")
        m4.metric("Lost money", f"{s['lost_money']:.0%}", help="Share that grossed less than their budget.")
        if s["films"] < 10:
            st.caption(f"Only {s['films']} films match, so these rates are not reliable.")
 
        st.plotly_chart(breakout_figure(data, max_budget, breakout_genre, multiple), width="stretch")
        st.caption(f"{len(tagged):,} films released {years[0]}–{years[1]} with reported box office "
                   "(films with no reported gross are always left out here). Hover a dot to see the film.")
 
        with st.expander(f"The {s['breakouts']} breakout films"):
            hits = tagged[tagged["selected"] & tagged["breakout"]].sort_values("multiple", ascending=False)
            st.dataframe(
                hits[["movie", "year", "genres", "production_budget", "worldwide_gross", "multiple"]],
                hide_index=True, width="stretch",
                column_config={
                    "movie": "Film", "year": st.column_config.NumberColumn("Year", format="%d"),
                    "genres": "Genres",
                    "production_budget": st.column_config.NumberColumn("Budget", format="dollar"),
                    "worldwide_gross": st.column_config.NumberColumn("Worldwide gross", format="dollar"),
                    "multiple": st.column_config.NumberColumn("Gross / budget", format="%.1fx"),
                })
 
# ===================== Release-month seasonality =====================
with tab_release_month:
    st.divider()
    st.subheader("When should you release it?")
    st.caption("When in the year do movies do best, and does each genre have its own best season? "
               "Pick a genre to see how its films performed by release month.")
 
    r1, r2 = st.columns(2)
    season_genre = r1.selectbox("Genre", [ALL_GENRES] + GENRES, key="season_genre",
                                help="A film counts toward every genre it is tagged with.")
    measure = r2.radio("Measure", list(SEASON_MEASURES), horizontal=True,
                       help="Typical return = median worldwide gross / budget. "
                            "Hit rate = share of films that grossed more than their budget.")
 
    films = season_films(data, season_genre)
    if films.empty:
        st.warning(f"No {season_genre} films with reported box office in {years[0]}–{years[1]}.")
    else:
        months = season_table(films)
        col = SEASON_MEASURES[measure]
        baseline = films["multiple"].median() if col == "median_multiple" else (films["multiple"] >= 1).mean()
        if col == "hit_rate":
            fmt, vs_year = (lambda v: f"{v:.0%}"), (lambda v: f"{(v - baseline) * 100:+.0f} pts vs. all year")
        else:
            fmt, vs_year = (lambda v: f"{v:.1f}x"), (lambda v: f"{v - baseline:+.1f}x vs. all year")
        trusted = months[months["reliable"]]
 
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Films", f"{len(films):,}")
        if len(trusted) < 2:
            st.caption(f"Fewer than two months have {SEASON_MIN_FILMS}+ films, "
                       "so there is no reliable best or worst month.")
        else:
            best, worst = trusted.loc[trusted[col].idxmax()], trusted.loc[trusted[col].idxmin()]
            m2.metric("Best month", f"{best['month']} · {fmt(best[col])}", vs_year(best[col]))
            m3.metric("Worst month", f"{worst['month']} · {fmt(worst[col])}", vs_year(worst[col]))
        busiest = months.loc[months["films"].idxmax()]
        m4.metric("Busiest month", busiest["month"], f"{busiest['films']} films released", delta_color="off",
                  help="Most crowded release month. Compare it with the best month: they are often not the same.")
 
        st.plotly_chart(season_figure(months, measure, baseline, season_genre), width="stretch")
        st.caption(f"Orange = better than this genre's year-round value, blue = worse. Gray months have fewer "
                   f"than {SEASON_MIN_FILMS} films and aren't used for best/worst. Films with no reported "
                   "gross are left out. Hover a bar for the top film that month.")
 
# ===================== Number of genres =====================
with tab_genre_count:
    st.divider()
    st.subheader("Does mixing genres pay off?")
    st.caption("For films tagged with several genres, does the number of genres matter, or only the "
               "primary (first-listed) genre? Switch the measure to compare budget, gross, and return.")
 
    all_count_films = genre_count_films(data, ALL_GENRES)
    primary_counts = all_count_films["primary_genre"].value_counts()
    primaries = [g for g in GENRES if primary_counts.get(g, 0) >= 30]   # enough films to split 5 ways
 
    g1, g2 = st.columns(2)
    primary = g1.selectbox("Primary genre", [ALL_GENRES] + primaries, key="count_primary",
                           help="The first genre a film is listed under. Only genres with 30+ films are shown.")
    count_measure = g2.radio("Measure", list(GENRE_COUNT_MEASURES), horizontal=True, key="count_measure")
 
    count_films = genre_count_films(data, primary)
    if count_films.empty:
        st.warning(f"No films with primary genre {primary} in {years[0]}–{years[1]}.")
    else:
        few = count_films[count_films["genre_count"].isin(["1", "2"])]
        many = count_films[count_films["genre_count"].isin(["4", "5+"])]
 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Films", f"{len(count_films):,}")
        if len(few) < GENRE_COUNT_MIN_FILMS or len(many) < GENRE_COUNT_MIN_FILMS:
            st.caption(f"Not enough films with both 1–2 and 4+ genres to compare them reliably.")
        else:
            few_budget, many_budget = few["production_budget"].median(), many["production_budget"].median()
            few_gross, many_gross = few["worldwide_gross"].median(), many["worldwide_gross"].median()
            few_return, many_return = few["multiple"].median(), many["multiple"].median()
            c2.metric("Budget, 4+ genres", fmt_money(many_budget),
                      f"{many_budget / few_budget:.1f}x vs. 1–2 genres", delta_color="off")
            c3.metric("Gross, 4+ genres", fmt_money(many_gross),
                      f"{many_gross / few_gross:.1f}x vs. 1–2 genres", delta_color="off")
            c4.metric("Return, 4+ genres", f"{many_return:.1f}x",
                      f"{many_return - few_return:+.1f}x vs. 1–2 genres",
                      help="Typical return = median worldwide gross / budget. Budget and gross deltas are ratios: 4.0x = four times as much.")
 
        st.plotly_chart(genre_count_figure(genre_count_table(count_films), count_measure, primary),
                        width="stretch")
        st.caption(f"Gray bars have fewer than {GENRE_COUNT_MIN_FILMS} films. Films with no reported gross "
                   "are left out. Hover a bar to see budget, gross, and return together.")
 
# ------------------------------------------------------------
# ROI Relationship Analysis
# ------------------------------------------------------------
 
with tab_roi:
    st.divider()
 
    st.subheader(
        "How Movie Characteristics Relate to ROI"
    )
 
    st.write(
        "Explore how different movie characteristics relate to ROI. "
        "Movies are divided into four approximately equal groups: "
        "Low, Medium, High, and Very High."
    )
 
    # --------------------------------------------------------
    # Choose movie characteristic
    # --------------------------------------------------------
 
    selected_name = st.selectbox(
        "Choose a movie characteristic",
        list(ROI_VARIABLES.keys()),
        key="roi_relationship_variable",
    )
 
    selected_variable = ROI_VARIABLES[
        selected_name
    ]
 
    # --------------------------------------------------------
    # NEW 1: Mean vs Median toggle
    # --------------------------------------------------------
 
    roi_statistic = st.radio(
        "Choose ROI summary statistic",
        ["Median ROI", "Mean ROI"],
        horizontal=True,
        key="roi_statistic_choice",
    )
 
    # --------------------------------------------------------
    # Create visualization
    # --------------------------------------------------------
 
    fig, correlation, summary, movie_count = (
        roi_relationship_figure(
            df,
            selected_variable,
            selected_name,
            roi_statistic,
        )
    )
 
    st.plotly_chart(
        fig,
        width="stretch",
    )
 
    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.metric(
            "Pearson correlation (r)",
            f"{correlation:.3f}",
        )
 
    with col2:
        st.metric(
            "Movies analyzed",
            f"{movie_count:,}",
        )
 
    # --------------------------------------------------------
    # NEW 2, 3, 4: Quick analysis
    # --------------------------------------------------------
 
    comparison = roi_comparison_metrics(
        summary,
        roi_statistic,
    )
 
    strongest_name, strongest_r = (
        strongest_roi_relationship(df)
    )
 
    st.subheader("Quick Analysis")
 
    col3, col4, col5 = st.columns(3)
 
    # Strongest relationship
    with col3:
        st.metric(
            "Strongest ROI Relationship",
            strongest_name,
            f"r = {strongest_r:.3f}",
        )
 
    # Highest ROI group
    with col4:
        st.metric(
            f"Highest {roi_statistic} Group",
            comparison["highest_group"],
            f"{comparison['highest_roi']:.2f} ROI",
        )
 
    # Low to Very High change
    with col5:
        st.metric(
            "Low → Very High ROI Change",
            f"{comparison['roi_change']:+.2f}",
        )
 
    # --------------------------------------------------------
    # Key Insight
    # --------------------------------------------------------
 
    st.subheader("Key Insight")
 
    insight = roi_key_insight(
        summary,
        correlation,
        selected_name,
        roi_statistic,
    )
 
    st.info(insight)
 
    # --------------------------------------------------------
    # Quartile ranges
    # --------------------------------------------------------
 
    st.subheader(
        f"{selected_name} Group Ranges"
    )
 
    st.write(
        "The ranges below show what Low, Medium, High, "
        "and Very High represent for the selected characteristic."
    )
 
    range_table = create_range_table(
        summary,
        selected_name,
    )
 
    st.dataframe(
        range_table,
        width="stretch",
        hide_index=True,
    )
 
    st.caption(
        "Each level contains approximately 25% of the movies. "
        "Hover over the graph to see the value range, median ROI, "
        "mean ROI, and number of movies in each group."
    )
 
    st.caption(
        "The Pearson correlation and strongest relationship "
        "are calculated using the movies remaining after the "
        "current Streamlit filters."
    )
 