"""
Random Forest model for the studio budget planner (TH).

Predicts three things for a film, using only what a studio knows before making it:
    1. Average performance: worldwide gross / budget        -> Random Forest regressor
    2. Bust:      grossed less than its budget               -> Random Forest classifier
    3. Breakout:  grossed at least 6x its budget             -> Random Forest classifier

A Random Forest trains hundreds of decision trees, each on a random sample of the films
and inputs, and averages their predictions ("bagging"). Averaging many independent trees
makes it hard for a few extreme films to throw the model off.

Trained on films released 2010-2016 and tested on 2017-2018 (films it never saw).
It runs twice, with the original columns and with the extra TMDB columns from
tmdb_enrich.py, and compares both against a baseline that uses no model at all.

Run from this folder (needs data/movie_roi_tmdb.csv from tmdb_enrich.py):
    python random_forest.py            # 2017-2018 test
    python random_forest.py --recent   # also test on 2022-2024 films (needs tmdb_recent.py)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, roc_auc_score

HERE = Path(__file__).parent
DATA = HERE / "data" / "movie_roi_tmdb.csv"
RECENT = HERE / "data" / "recent_films_tmdb.csv"   # from tmdb_recent.py
RESULTS = HERE / "data" / "rf_results.csv"

TRAIN_YEARS, TEST_YEARS = (2010, 2016), (2017, 2018)
BREAKOUT_MULTIPLE = 6
GENRE_COLS = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama",
    "Family", "Fantasy", "History", "Horror", "Music", "Mystery", "Romance",
    "Science Fiction", "Thriller", "War", "Western",
]
RATINGS = ["G", "PG", "PG-13", "R"]
SOURCE_FLAGS = ["based_on_book", "based_on_comic", "based_on_true_story", "based_on_video_game", "is_remake"]
SEED = 42
FOREST = dict(n_estimators=500, min_samples_leaf=5, random_state=SEED, n_jobs=-1)   # 500 trees


# ---------------------------------------------------------------- data
def load_films(path=DATA, years=(TRAIN_YEARS[0], TEST_YEARS[1])) -> pd.DataFrame:
    """Films in the year range with a budget and reported box office, plus the three targets."""
    df = pd.read_csv(path)
    df = df[df["year"].between(*years)
            & (df["production_budget"] > 0) & ~df["no_reported_box_office"].astype(bool)].copy()
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["multiple"] = df["worldwide_gross"] / df["production_budget"]
    df["log_multiple"] = np.log(df["multiple"].clip(lower=0.01))   # a few films grossed ~$0
    df["bust"] = (df["multiple"] < 1).astype(int)
    df["breakout"] = (df["multiple"] >= BREAKOUT_MULTIPLE).astype(int)
    df["split"] = np.where(df["year"] <= TRAIN_YEARS[1], "train", "test")
    return df.sort_values("release_date").reset_index(drop=True)


def load_with_recent() -> pd.DataFrame:
    """2010-2018 films as training data and 2022-2024 films (tmdb_recent.py) as the test set."""
    old = load_films()
    recent = load_films(RECENT, years=(2022, 2024))
    old["split"], recent["split"] = "train", "test"
    df = pd.concat([old, recent], ignore_index=True)
    for g in GENRE_COLS:
        df[g] = df[g].fillna(0).astype(int)
    return df.sort_values("release_date").reset_index(drop=True)


def track_record(df: pd.DataFrame, people_col: str, prefix: str, max_people: int = 3) -> pd.DataFrame:
    """For each film, the earlier films (in this dataset) of its director / top cast:
    how many there were and their average log return. Only films released *before*
    this one count, so the model never sees results from the future."""
    history: dict[str, list[float]] = {}
    counts, means = [], []
    for _, row in df.iterrows():
        people = [p for p in str(row.get(people_col) or "").split("|") if p and p != "nan"][:max_people]
        past = [v for p in people for v in history.get(p, [])]
        counts.append(len(past))
        means.append(np.mean(past) if past else np.nan)
        for p in people:   # add this film after reading, so it only helps later films
            history.setdefault(p, []).append(row["log_multiple"])
    return pd.DataFrame({f"{prefix}_prior_films": counts, f"{prefix}_prior_return": means}, index=df.index)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (feature table, original columns, TMDB columns). Everything is pre-release information."""
    X = pd.DataFrame(index=df.index)
    # Original columns
    X["log_budget"] = np.log(df["production_budget"])
    X["year"] = df["year"]
    X["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)     # month as a circle: Dec is next to Jan
    X["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    X["english"] = (df["original_language"] == "en").astype(int)
    X["genre_count"] = df["genres"].str.split(",").str.len().fillna(0)   # a few films have no genres listed
    for g in GENRE_COLS:
        X[f"genre_{g}"] = df[g]
    base = list(X.columns)

    # New TMDB columns
    X["runtime"] = df["runtime"].fillna(df["runtime"].median())
    for r in RATINGS:
        X[f"rated_{r}"] = (df["us_certification"] == r).astype(int)
    X["is_sequel"] = df["is_sequel"].eq(True).astype(int)
    X["earlier_franchise_films"] = df["franchise_earlier_films"].fillna(0).clip(upper=10)
    for f in SOURCE_FLAGS:
        X[f] = df[f].eq(True).astype(int)
    train_mean = df.loc[df["split"] == "train", "log_multiple"].mean()
    for people_col, prefix in [("director", "director"), ("top_cast", "cast")]:
        rec = track_record(df, people_col, prefix)
        X[f"{prefix}_prior_films"] = rec[f"{prefix}_prior_films"]
        # No earlier films -> use the training average, so "unknown" sits in the middle
        X[f"{prefix}_prior_return"] = rec[f"{prefix}_prior_return"].fillna(train_mean)
    tmdb = [c for c in X.columns if c not in base]
    return X, base, tmdb


# ---------------------------------------------------------------- model
TARGETS = [
    ("log_multiple", "Average performance", "Typical miss (lower is better)"),
    ("bust", "Bust (< 1x budget)", "AUC (higher is better)"),
    ("breakout", f"Breakout ({BREAKOUT_MULTIPLE}x+ budget)", "AUC (higher is better)"),
]


def fit_forest(target: str, X_train: pd.DataFrame, y_train: pd.Series):
    model = RandomForestRegressor(**FOREST) if target == "log_multiple" else RandomForestClassifier(**FOREST)
    return model.fit(X_train, y_train)


def score(target: str, y_true: pd.Series, pred: np.ndarray) -> float:
    """Average performance: typical miss as a factor (2.7 = typically off by 2.7x either way).
    Bust / breakout: AUC (0.5 = coin flip, 1.0 = perfect ranking)."""
    if target == "log_multiple":
        return float(np.exp(mean_absolute_error(y_true, pred)))
    return float(roc_auc_score(y_true, pred))


def evaluate(df: pd.DataFrame, X: pd.DataFrame, base: list[str], tmdb: list[str]) -> pd.DataFrame:
    """One row per prediction: baseline vs. Random Forest with original and with original + TMDB columns."""
    train, test = df["split"] == "train", df["split"] == "test"
    rows = []
    for target, label, measure in TARGETS:
        y = df[target]
        if target == "log_multiple":   # baseline: predict the typical (median) film for everyone
            baseline = score(target, y[test], np.full(test.sum(), y[train].median()))
        else:                          # baseline: same odds for every film = coin flip
            baseline = 0.5
        row = {"Prediction": label, "Measure": measure, "Baseline (no model)": baseline}
        for name, cols in [("Random Forest (original)", base), ("Random Forest (original + TMDB)", base + tmdb)]:
            model = fit_forest(target, X.loc[train, cols], y[train])
            pred = (model.predict(X.loc[test, cols]) if target == "log_multiple"
                    else model.predict_proba(X.loc[test, cols])[:, 1])
            row[name] = score(target, y[test], pred)
        rows.append(row)
    return pd.DataFrame(rows)


def top_drivers(df: pd.DataFrame, X: pd.DataFrame, cols: list[str], target: str, n: int = 8) -> pd.Series:
    """Which inputs the forest relies on most, measured on the test films (permutation importance:
    how much the AUC drops when that column is scrambled)."""
    train, test = df["split"] == "train", df["split"] == "test"
    model = fit_forest(target, X.loc[train, cols], df.loc[train, target])
    imp = permutation_importance(model, X.loc[test, cols], df.loc[test, target], scoring="roc_auc",
                                 n_repeats=20, random_state=SEED)
    return pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False).head(n)


def run(df: pd.DataFrame, title: str) -> pd.DataFrame:
    X, base, tmdb = build_features(df)
    n_train, n_test = (df["split"] == "train").sum(), (df["split"] == "test").sum()
    print(f"=== {title}: trained on {n_train:,} films, tested on {n_test:,}")
    print(f"Original inputs: {len(base)} | TMDB inputs added: {len(tmdb)}\n")
    results = evaluate(df, X, base, tmdb)
    shown = results.copy()
    miss = shown["Prediction"] == "Average performance"
    for col in shown.columns[2:]:
        shown[col] = [f"{v:.2f}x" if m else f"{v:.2f}" for v, m in zip(shown[col], miss)]
    print(shown.to_string(index=False), "\n")
    return results.assign(test_set=title)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recent", action="store_true", help="also test on 2022-2024 films")
    args = parser.parse_args()

    df = load_films()
    all_results = [run(df, "Test on 2017-2018")]

    X, base, tmdb = build_features(df)
    for target in ["bust", "breakout"]:
        print(f"What the forest relies on most for {target} (AUC drop when scrambled):")
        print(top_drivers(df, X, base + tmdb, target).round(3).to_string(), "\n")

    if args.recent and RECENT.exists():
        all_results.append(run(load_with_recent(), "Test on 2022-2024"))

    pd.concat(all_results).to_csv(RESULTS, index=False)
    print(f"Saved results to {RESULTS.relative_to(HERE)}")


if __name__ == "__main__":
    main()
