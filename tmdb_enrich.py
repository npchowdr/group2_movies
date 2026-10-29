"""
Enrich movie_roi.csv with extra pre-release information from the TMDB API.

Adds, for each film: runtime, US age rating, franchise info (and whether the film
was a sequel *at the time it came out*), director, top cast, production companies,
and "based on" keywords (novel, comic, true story, remake...).

Setup (one time):
    1. Make a free account at https://www.themoviedb.org and request an API key
       (Settings -> API). Either the "API Key" or the "API Read Access Token" works.
    2. Put it in a file named .env in this folder (already in .gitignore):
           TMDB_API_KEY=your_key_here

Run from this folder:
    python tmdb_enrich.py --limit 20     # quick test on 20 films
    python tmdb_enrich.py                # all films

Output: data/movie_roi_tmdb.csv (the data folder is git-ignored). Every API response
is cached in data/tmdb_cache/, so re-running only downloads what's missing.
Check the `match_status` column: rows marked "check" or "none" need a manual look.
"""

import argparse
import json
import math
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
SOURCE = HERE / "movie_roi.csv"
OUT_DIR = HERE / "data"
CACHE_DIR = OUT_DIR / "tmdb_cache"
OUTPUT = OUT_DIR / "movie_roi_tmdb.csv"
API = "https://api.themoviedb.org/3"

SOURCE_KEYWORDS = {   # TMDB keyword -> our column
    "based on novel or book": "based_on_book",
    "based on comic": "based_on_comic",
    "based on true story": "based_on_true_story",
    "biography": "based_on_true_story",
    "remake": "is_remake",
    "based on video game": "based_on_video_game",
    "based on young adult novel": "based_on_book",
}


# ---------------------------------------------------------------- API access
def read_api_key() -> str:
    key = os.environ.get("TMDB_API_KEY")
    env_file = HERE / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():   # -sig: ignore a Notepad BOM
            if line.strip().startswith("TMDB_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No TMDB key found. Put TMDB_API_KEY=... in a .env file next to this script.")
    return key


class TMDB:
    """Tiny TMDB client with an on-disk cache and polite rate limiting."""

    def __init__(self, key: str):
        self.session = requests.Session()
        if key.startswith("eyJ"):            # v4 "API Read Access Token"
            self.session.headers["Authorization"] = f"Bearer {key}"
            self.params = {}
        else:                                # v3 "API Key"
            self.params = {"api_key": key}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, path: str, **params) -> dict:
        name = re.sub(r"[^A-Za-z0-9]+", "_", path + "_" + "_".join(f"{k}-{v}" for k, v in sorted(params.items())))
        cache_file = CACHE_DIR / f"{name[:180]}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

        for attempt in range(5):
            resp = self.session.get(f"{API}{path}", params={**self.params, **params}, timeout=20)
            if resp.status_code == 429:      # rate limited: wait and retry
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 401:
                raise SystemExit("TMDB rejected the API key (401). Check the value in .env.")
            resp.raise_for_status()
            data = resp.json()
            cache_file.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.05)
            return data
        raise RuntimeError(f"Gave up on {path} after repeated rate limiting")


# ---------------------------------------------------------------- matching
# Hand-checked fixes for films the automatic matching gets wrong: {title in movie_roi.csv: TMDB id or None}
MANUAL_MATCHES: dict[str, int | None] = {
    "Snabba Cash": 29920,                  # Easy Money (2010), US release 2012; search picked its sequel
    "Fireflies in the Garden": 7353,       # correct film: 2008 festival premiere, US release 2011
    "All the Boys Love Mandy Lane": 9022,  # correct film: 2006 festival premiere, US release 2013
    # The original movie_roi.csv row itself looks mismatched (its votes fit a short film, its
    # budget a feature), so we can't tell which film the box office belongs to: leave blank.
    "Redemption": None,
    "Test": None,
}

# TMDB stores a film's first release anywhere (often a premiere), our CSV has the US date
DATE_TOLERANCE_DAYS = 14


def days_apart(a: str | None, b: str) -> float:
    try:
        return abs((pd.Timestamp(a) - pd.Timestamp(b)).days)
    except (ValueError, TypeError):
        return float("inf")


def clean_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(t).lower().replace("&", "and")).strip()


def vote_similarity(ours, theirs) -> float:
    """1.0 when TMDB's vote count is close to the one in our CSV, falling to 0 at 10x apart.
    Our CSV came from TMDB, so this is a strong fingerprint for the right film."""
    try:
        ratio = (float(ours) + 20) / (float(theirs) + 20)   # +20 keeps tiny counts from swinging wildly
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, 1 - abs(math.log10(ratio)))


def score_candidate(c: dict, title: str, release_date: str, vote_count, language) -> dict:
    title_sim = max(SequenceMatcher(None, clean_title(title), clean_title(c.get(k, ""))).ratio()
                    for k in ("title", "original_title"))
    gap = days_apart(c.get("release_date"), release_date)
    date_sim = 1.0 if gap <= DATE_TOLERANCE_DAYS else 0.8 if gap <= 365 else 0.5 if gap <= 3 * 365 else 0.0
    votes = vote_similarity(vote_count, c.get("vote_count"))
    same_lang = 1.0 if c.get("original_language") == language else 0.0
    score = 0.5 * title_sim + 0.2 * date_sim + 0.2 * votes + 0.1 * same_lang
    return {"title": title_sim, "date": date_sim, "votes": votes, "lang": same_lang, "score": score}


def find_match(tmdb: TMDB, title: str, year: int, release_date: str,
               vote_count=None, language=None) -> tuple[dict | None, float, str]:
    """Search TMDB by title and pick the best candidate using title, release date,
    vote count, and original language. Returns (result, score 0-1, status)."""
    if title in MANUAL_MATCHES:                       # hand-checked fixes
        movie_id = MANUAL_MATCHES[title]
        if movie_id is None:
            return None, 0.0, "none"
        return {"id": movie_id}, 1.0, "manual"

    candidates = {}
    for params in ({"primary_release_year": year}, {}):   # the release year, then any year
        for c in tmdb.get("/search/movie", query=title, include_adult="false", **params).get("results", []):
            candidates[c["id"]] = c

    scored = [(score_candidate(c, title, release_date, vote_count, language), c) for c in candidates.values()]
    if not scored:
        return None, 0.0, "none"
    s, best = max(scored, key=lambda sc: sc[0]["score"])

    title_ok = s["title"] >= 0.95
    if title_ok and s["date"] >= 0.8 and s["votes"] >= 0.5:
        status = "exact"
    elif title_ok and s["votes"] >= 0.7 and s["lang"] == 1.0:   # e.g. festival premiere years before US release
        status = "exact"
    elif s["score"] >= 0.8:
        status = "likely"
    else:
        status = "check"
    return best, round(s["score"], 3), status


# ---------------------------------------------------------------- details
def us_certification(details: dict) -> str | None:
    for country in details.get("release_dates", {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            certs = [r.get("certification") for r in country.get("release_dates", []) if r.get("certification")]
            if certs:
                return certs[0]
    return None


def sequel_info(tmdb: TMDB, details: dict) -> tuple[str | None, int | None]:
    """Franchise name and how many earlier films in it came out before this one.
    Uses release order, so the first film of a franchise counts as 0 (not a sequel yet),
    which avoids leaking "it later got sequels because it was a hit" into the model."""
    coll = details.get("belongs_to_collection")
    if not coll:
        return None, None
    parts = tmdb.get(f"/collection/{coll['id']}").get("parts", [])
    this_date = details.get("release_date") or ""
    earlier = [p for p in parts if p.get("id") != details.get("id")       # skip the film itself
               and p.get("release_date") and p["release_date"] < this_date]
    return coll.get("name"), len(earlier)


def extract_features(tmdb: TMDB, movie_id: int) -> dict:
    d = tmdb.get(f"/movie/{movie_id}", append_to_response="release_dates,credits,keywords")
    crew = d.get("credits", {}).get("crew", [])
    cast = sorted(d.get("credits", {}).get("cast", []), key=lambda c: c.get("order", 99))
    keywords = [k["name"] for k in d.get("keywords", {}).get("keywords", [])]
    franchise, earlier_films = sequel_info(tmdb, d)

    row = {
        "tmdb_id": d.get("id"),
        "tmdb_title": d.get("title"),
        "tmdb_release_date": d.get("release_date"),
        "runtime": d.get("runtime") or None,
        "us_certification": us_certification(d),
        "franchise": franchise,
        "franchise_earlier_films": earlier_films,
        "is_sequel": bool(earlier_films) if earlier_films is not None else False,
        "director": "|".join(c["name"] for c in crew if c.get("job") == "Director"),
        "top_cast": "|".join(c["name"] for c in cast[:3]),
        "production_companies": "|".join(c["name"] for c in d.get("production_companies", [])[:3]),
        "origin_country": "|".join(d.get("origin_country", []) or
                                   [c["iso_3166_1"] for c in d.get("production_countries", [])]),
        "keywords": "|".join(keywords),
    }
    for col in set(SOURCE_KEYWORDS.values()):
        row[col] = False
    for kw in keywords:
        if kw in SOURCE_KEYWORDS:
            row[SOURCE_KEYWORDS[kw]] = True
    return row


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process the first N films (for testing)")
    args = parser.parse_args()

    tmdb = TMDB(read_api_key())
    films = pd.read_csv(SOURCE)
    if args.limit:
        films = films.head(args.limit)

    rows = []
    for i, f in enumerate(films.itertuples(index=False), start=1):
        release_date = pd.to_datetime(f.release_date).strftime("%Y-%m-%d")
        row = {"match_key": f.match_key}
        try:
            match, score, status = find_match(tmdb, f.movie, int(f.year), release_date,
                                              f.vote_count, f.original_language)
            row.update(match_score=score, match_status=status)
            if match is not None:
                row.update(extract_features(tmdb, match["id"]))
        except requests.RequestException as e:
            row.update(match_status="error", error=str(e)[:200])
        rows.append(row)
        if i % 50 == 0 or i == len(films):
            print(f"{i}/{len(films)} films processed")

    extra = pd.DataFrame(rows)
    out = films.merge(extra, on="match_key", how="left")
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUTPUT, index=False)

    print(f"\nSaved {OUTPUT.relative_to(HERE)}")
    print(out["match_status"].value_counts().to_string())
    print("Rows marked 'check' or 'none' are worth a manual look.")


if __name__ == "__main__":
    main()
