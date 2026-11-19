# Movie Budgets & ROI by Genre

A Streamlit app for exploring which kinds of movies pay off, which are risky, and how a studio could split a year's production budget, using 1,547 films released 2010-2018 (budgets, box office, and extra details from TMDB).

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

The data files are committed, so there is no setup step beyond installing the dependencies (streamlit, plotly, pandas, numpy, scikit-learn). If Windows blocks the `streamlit` command, use `python -m streamlit run app.py` instead. The TMDB API key (`.env`) is only needed to re-run `tmdb_enrich.py` or `tmdb_recent.py`.

## What's in the app

| Tab | What it does |
| --- | --- |
| Outcome predictor | Describe a film idea (budget, genres, release month, rating, runtime, sequel, source) and see its chance of a bust, middle result or breakout, plus its typical and expected ROI. |
| Budget planner | Enter an annual budget and a risk level (Conservative, Balanced or Aggressive). Recommends a mix of film types and simulates the year 4,000 times using real films of each type. |
| Budgets | Budget distribution by genre, summary stats, and average budget with standard deviation. |
| Risk & reward | Mean ROI vs. volatility by genre, the biggest blockbuster and worst bust in each genre, and how ROI relates to budget, grosses, popularity and votes. |
| Breakouts & timing | A low-budget breakout finder, the best release months, and whether mixing genres pays off. |

The sidebar filters (release years, exclude films with no box office) apply to the analysis tabs. The Outcome predictor and Budget planner always use every 2010-2018 film.

## Repo layout

| File | What it does |
| --- | --- |
| `app.py` | The app. All five tabs. |
| `apputil.py` | Data loading, charts, the Outcome predictor model, and the Budget planner logic. No Streamlit calls, so it can be tested on its own. |
| `random_forest.py` | Random forest for bust, breakout and average performance, with its own tests (`python random_forest.py`). |
| `tmdb_enrich.py` | Adds runtime, rating, franchise, cast, crew and keyword data from TMDB to `movie_roi.csv`. |
| `tmdb_recent.py` | Pulls 2022-2024 films from TMDB as a second test set. |
| `movie_roi.csv` | The base dataset: 1,602 films (1946-2019); the app uses the 1,547 released 2010-2018. |
| `data/movie_roi_tmdb.csv` | The same films with the TMDB columns added. |
| `data/recent_films_tmdb.csv` | 534 films from 2022-2024, used only as a test set. |
| `black_background_image.jpg` | The app's background image. |

Columns in `movie_roi.csv`: movie, year, month, release_date, production_budget, domestic_gross, foreign_gross, worldwide_gross, profit, profit_margin, roi, pct_foreign, popularity, vote_average, vote_count, original_language, genres, one 0/1 column per genre (Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western), and no_reported_box_office. ROI is profit divided by budget.

`data/movie_roi_tmdb.csv` adds: runtime, us_certification, franchise, franchise_earlier_films, is_sequel, director, top_cast, production_companies, origin_country, keywords, based_on_book, based_on_true_story, based_on_comic, based_on_video_game, is_remake, and the match columns (match_score, match_status, tmdb_id, tmdb_title, tmdb_release_date) that link each film to TMDB.

## How the Outcome predictor works

- **Outcomes:** Bust is ROI below 0 (the film lost money). Breakout is ROI of 5 or more (it grossed 6x its budget or more). Middle is everything in between.
- **Inputs, all known before release:** budget, genres, release month, US rating, runtime, sequel or franchise, and source material (book, comic, true story, video game, remake).
- **Model:** the average of a logistic regression and a random forest, trained on the 1,450 films that have a usable ROI.
- **ROI numbers:** Typical ROI is what a middle-of-the-road film like this earns. Expected ROI is the average return, each outcome's chance times that outcome's average ROI, with hits capped at +3,000% so one mega-hit can't distort it. Typical ROI is the steadier number, because a few blockbusters swing the average.
- **Testing:** for each year from 2014 to 2018, the model trained only on earlier films and predicted that year. It ranked busts better than a budget-and-genre-only model (AUC 0.73 vs 0.69) and gave better probabilities than a flat guess (log loss 0.863 vs 0.971). Typical ROI missed by 3.21 on average, compared with 3.34 for always guessing the median.
- **Limits:** these are rough odds, not predictions. The model doesn't use cast, director or marketing spend, and it only covers 2010-2018.

## Where the data comes from

- `movie_roi.csv`: production budgets and box office grosses for about 1,600 films.
- `data/movie_roi_tmdb.csv`: the same films enriched with runtime, age rating, franchise, cast and crew, and keyword data from [The Movie Database (TMDB)](https://www.themoviedb.org/), using `tmdb_enrich.py`.

This product uses the TMDB API but is not endorsed or certified by TMDB.
