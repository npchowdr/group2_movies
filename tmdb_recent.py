"""
Pull recent films (default 2022-2024) from TMDB as a second, post-pandemic test set.

For each year it takes the highest-grossing films TMDB knows about, keeps the ones
with a reported budget and revenue, and builds the same columns as movie_roi.csv
plus the extra columns from tmdb_enrich.py, so the models can be tested on them.

Note: TMDB's budget and revenue figures are entered by contributors. They're good
for big releases but less reliable for small films, so treat results as a check on
whether the 2010s patterns still hold, not as exact numbers.

Uses the same .env key as tmdb_enrich.py. Run from this folder:
    python tmdb_recent.py                        # 2022-2024
    python tmdb_recent.py --years 2023 --pages 2 # quick test

Output: data/recent_films_tmdb.csv (git-ignored)
"""

import argparse

import pandas as pd

from tmdb_enrich import OUT_DIR, TMDB, extract_features, read_api_key

OUTPUT = OUT_DIR / "recent_films_tmdb.csv"
MIN_BUDGET = 1_000_000   # smaller "budgets" on TMDB are often placeholders


def recent_films(tmdb: TMDB, year: int, pages: int) -> list[dict]:
    """Top films of a year by revenue (20 per page)."""
    films = []
    for page in range(1, pages + 1):
        data = tmdb.get("/discover/movie", primary_release_year=year, sort_by="revenue.desc",
                        include_adult="false", page=page)
        films += data.get("results", [])
        if page >= data.get("total_pages", 0):
            break
    return films


def film_row(tmdb: TMDB, movie_id: int, genre_names: dict[int, str]) -> dict | None:
    d = tmdb.get(f"/movie/{movie_id}", append_to_response="release_dates,credits,keywords")
    budget, revenue = d.get("budget") or 0, d.get("revenue") or 0
    if budget < MIN_BUDGET or revenue <= 0 or not d.get("release_date"):
        return None
    release = pd.Timestamp(d["release_date"])
    genres = [g["name"] for g in d.get("genres", [])]
    row = {
        "movie": d.get("title"), "year": release.year, "month": release.month,
        "release_date": release.strftime("%m/%d/%Y"),
        "production_budget": budget, "worldwide_gross": revenue,
        "profit": revenue - budget, "roi": (revenue - budget) / budget,
        "original_language": d.get("original_language"),
        "popularity": d.get("popularity"), "vote_average": d.get("vote_average"),
        "vote_count": d.get("vote_count"),
        "genres": ",".join(genres), "no_reported_box_office": False,
        "match_key": f"{release.year} {d.get('title')}", "match_status": "discover",
    }
    for name in genre_names.values():
        row[name] = int(name in genres)
    row.update(extract_features(tmdb, movie_id))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--pages", type=int, default=15, help="pages of 20 films per year (default 15 = top 300)")
    args = parser.parse_args()

    tmdb = TMDB(read_api_key())
    genre_names = {g["id"]: g["name"] for g in tmdb.get("/genre/movie/list").get("genres", [])}

    rows = []
    for year in args.years:
        candidates = recent_films(tmdb, year, args.pages)
        kept = [r for r in (film_row(tmdb, c["id"], genre_names) for c in candidates) if r]
        print(f"{year}: {len(candidates)} films checked, {len(kept)} with budget and revenue")
        rows += kept

    out = pd.DataFrame(rows).drop_duplicates("tmdb_id")
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(out)} films to {OUTPUT.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
