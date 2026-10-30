"""Data pipeline: download match data -> clean -> load into SQL (Postgres or SQLite).

Leagues:
  EPL  English Premier League, 2005/06 -> today (football-data.co.uk)
  EGY  Egyptian Premier League, 2009/10 -> today (data/egypt_results.csv from fetch_egypt.py),
       falling back to 2010/11 -> 2017/18 (Kaggle: mahmoudelshabrawy/egyptianleague)
"""
import io
import os
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{Path(__file__).with_name('football.db').as_posix()}")
EPL_SEASONS = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(2005, 2027)]  # 0506 ... 2627
EPL_SOURCE = "https://football-data.co.uk/mmz4281/{season}/E0.csv"
EGY_DATASET = "mahmoudelshabrawy/egyptianleague"

LEAGUES = {"EPL": "English Premier League", "EGY": "Egyptian Premier League"}

EPL_COLUMNS = {
    "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HTHG": "home_ht_goals", "HTAG": "away_ht_goals", "Referee": "referee",
    "HxG": "home_xg", "AxG": "away_xg",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners", "HY": "home_yellow", "AY": "away_yellow",
    "HR": "home_red", "AR": "away_red",
}
ALL_COLUMNS = list(EPL_COLUMNS.values())


def engine():
    return create_engine(DATABASE_URL)


def _standardise(df: pd.DataFrame, league: str) -> pd.DataFrame:
    for col in ALL_COLUMNS:  # older seasons / other leagues miss some stats
        if col not in df.columns:
            df[col] = None
    df["league"] = league
    return df[["league", "season", *ALL_COLUMNS]]


def fetch_epl_season(season: str) -> pd.DataFrame:
    resp = requests.get(EPL_SOURCE.format(season=season), timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig", errors="replace")), on_bad_lines="skip")
    df = df[[c for c in EPL_COLUMNS if c in df.columns]].rename(columns=EPL_COLUMNS)
    df = df.dropna(subset=["home", "away", "home_goals"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed").dt.date
    df["season"] = season
    return _standardise(df, "EPL")


EGY_RECENT = Path(__file__).with_name("data") / "egypt_results.csv"  # made by fetch_egypt.py


def fetch_egypt_recent() -> pd.DataFrame:
    """2009/10 -> today, from fetch_egypt.py (results and goals only)."""
    raw = pd.read_csv(EGY_RECENT, parse_dates=["date"])
    first_year = raw.season_slug.str.extract(r"(\d{4})-")[0].astype(int)
    df = raw.assign(
        date=raw.date.dt.date,
        season=first_year.map(lambda y: f"{y % 100:02d}{(y + 1) % 100:02d}"),
        result=["H" if h > a else "A" if a > h else "D" for h, a in zip(raw.home_goals, raw.away_goals)],
    )
    return _standardise(df, "EGY")


def fetch_egypt() -> pd.DataFrame:
    if EGY_RECENT.exists():
        return fetch_egypt_recent()
    import kagglehub  # fallback: public 2010-2018 dataset, no account needed

    folder = Path(kagglehub.dataset_download(EGY_DATASET))
    raw = pd.read_csv(folder / "EgyptianLeague.csv")
    half_time = raw["Half Time Result"].str.extract(r"(\d+)\s*:\s*(\d+)").astype(float)
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], format="%a, %b %d, %Y").dt.date,
        "home": raw["Home Team"].str.strip(), "away": raw["Away Team"].str.strip(),
        "home_goals": raw["Home Goals"], "away_goals": raw["Away Goals"],
        "result": raw["Winner"].map({"Home": "H", "Draw": "D", "Away": "A"}),
        "home_ht_goals": half_time[0], "away_ht_goals": half_time[1],
        "referee": raw["Refree"],
        "home_yellow": raw["Yellow Home"], "away_yellow": raw["Yellow Away"],
        "home_red": raw["Red Home"] + raw["2nd Yellow Home"], "away_red": raw["Red Away"] + raw["2nd Yellow Away"],
        "season": raw["Season"].map(lambda y: f"{y % 100:02d}{(y + 1) % 100:02d}"),
    })
    return _standardise(df, "EGY")


def match_document(r) -> str:
    """Plain-text description of one match, used as a RAG document."""
    pair = lambda a, b: f"{int(a)}-{int(b)}" if pd.notna(a) and pd.notna(b) else None
    parts = [f"{r.date} {r.home} {int(r.home_goals)}-{int(r.away_goals)} {r.away}"]
    for label, a, b in (
        ("half-time", r.home_ht_goals, r.away_ht_goals), ("shots", r.home_shots, r.away_shots),
        ("on target", r.home_sot, r.away_sot), ("corners", r.home_corners, r.away_corners),
        ("yellow cards", r.home_yellow, r.away_yellow), ("red cards", r.home_red, r.away_red),
    ):
        if pair(a, b):
            parts.append(f"{label} {pair(a, b)}")
    if pd.notna(r.home_xg):
        parts.append(f"xG {r.home_xg:.2f}-{r.away_xg:.2f}")
    if pd.notna(r.referee):
        parts.append(f"referee {r.referee}")
    return ", ".join(parts) + "."


def run():
    frames = []
    for season in EPL_SEASONS:
        try:
            frames.append(fetch_epl_season(season))
        except requests.HTTPError as e:
            print(f"skip EPL {season}: {e}")
    frames.append(fetch_egypt())
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "league"]).reset_index(drop=True)
    df.insert(0, "match_id", range(1, len(df) + 1))
    docs = pd.DataFrame({
        "match_id": df.match_id, "league": df.league, "date": df.date, "home": df.home, "away": df.away,
        "content": [match_document(r) for r in df.itertuples()],
    })
    eng = engine()
    df.to_sql("matches", eng, if_exists="replace", index=False)
    docs.to_sql("documents", eng, if_exists="replace", index=False)
    with eng.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_matches_teams ON matches (home, away, date)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_matches_league ON matches (league, season, date)"))
    for league, part in df.groupby("league"):
        print(f"{league}: {len(part)} matches ({part.date.min()} -> {part.date.max()})")
    print(f"Saved to {DATABASE_URL}")


if __name__ == "__main__":
    run()
