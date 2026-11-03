"""Data pipeline: download match data -> clean -> load into SQL (Postgres or SQLite).

Leagues:
  EPL  English Premier League, 2005/06 -> today (football-data.co.uk)
  EGY  Egyptian Premier League, 2009/10 -> today (data/egypt_results.csv from fetch_egypt.py),
       falling back to 2010/11 -> 2017/18 (Kaggle: mahmoudelshabrawy/egyptianleague)
"""
import io
import os
import time
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


def _get(url: str) -> requests.Response:
    """GET with a few retries: football-data.co.uk sometimes drops the connection mid-run."""
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def _standardise(df: pd.DataFrame, league: str) -> pd.DataFrame:
    for col in ALL_COLUMNS:  # older seasons / other leagues miss some stats
        if col not in df.columns:
            df[col] = None
    df["league"] = league
    return df[["league", "season", *ALL_COLUMNS]]


def fetch_epl_season(season: str) -> pd.DataFrame:
    resp = _get(EPL_SOURCE.format(season=season))
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig", errors="replace")), on_bad_lines="skip")
    df = df[[c for c in EPL_COLUMNS if c in df.columns]].rename(columns=EPL_COLUMNS)
    df = df.dropna(subset=["home", "away", "home_goals"])
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="mixed").dt.date
    df["season"] = season
    return _standardise(df, "EPL")


EPL_FIXTURES = "https://football-data.co.uk/fixtures.csv"  # next round only, and empty for days after midweek
# full-season schedule (public domain); its dates matched all 90 played 2026/27 matches when checked on 3 Nov 2026
EPL_SCHEDULE = "https://raw.githubusercontent.com/openfootball/football.json/master/{start}-{end:02d}/en.1.json"
OPENFOOTBALL_NAMES = {
    "AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford", "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry", "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton",
    "Fulham FC": "Fulham", "Hull City AFC": "Hull", "Ipswich Town FC": "Ipswich", "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest", "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
}


def fetch_epl_fixtures(played: pd.DataFrame) -> pd.DataFrame:
    """Every EPL match not played yet: openfootball's season schedule, with football-data's dates for the
    next round when it lists them (those include TV moves)."""
    season = EPL_SEASONS[-1]
    start = 2000 + int(season[:2])
    games = _get(EPL_SCHEDULE.format(start=start, end=(start + 1) % 100)).json()["matches"]
    unknown = {t for m in games for t in (m["team1"], m["team2"])} - set(OPENFOOTBALL_NAMES)
    if unknown:
        raise SystemExit(f"Add these openfootball team names to OPENFOOTBALL_NAMES: {sorted(unknown)}")
    df = pd.DataFrame({"date": pd.to_datetime([m["date"] for m in games]).date,
                       "home": [OPENFOOTBALL_NAMES[m["team1"]] for m in games],
                       "away": [OPENFOOTBALL_NAMES[m["team2"]] for m in games]})
    done = set(zip(played.home, played.away))  # each pairing is played once per season
    df = df[[(h, a) not in done for h, a in zip(df.home, df.away)]]

    nxt = pd.read_csv(io.StringIO(_get(EPL_FIXTURES).content.decode("utf-8-sig", errors="replace")))
    nxt = nxt[nxt.Div == "E0"]
    exact = {(h, a): pd.to_datetime(d, dayfirst=True).date() for d, h, a in zip(nxt.Date, nxt.HomeTeam, nxt.AwayTeam)}
    df["date"] = [exact.get((h, a), d) for d, h, a in zip(df.date, df.home, df.away)]
    return df.assign(league="EPL", season=season)[["league", "season", "date", "home", "away"]].sort_values("date")


EGY_RECENT = Path(__file__).with_name("data") / "egypt_results.csv"  # made by fetch_egypt.py


EGY_KOOORA = Path(__file__).with_name("data") / "kooora_egypt.csv"  # made by fetch_kooora.py
# Kooora (Arabic) -> footballdatabase (English) names, matched on date + score over the 2026/27 season
KOOORA_NAMES = {
    "أبو قير للأسمدة": "Abou Qir Fertilizers SC", "الأهلي": "Al Ahly", "الاتحاد": "Al-Ittihad Alexandria",
    "البنك الأهلي": "National Bank", "الجونة": "El Gounah", "الجيش": "El Geish", "الزمالك": "El Zamalek",
    "السويس بتروجيت": "Petrojet", "الشرقية إنبي": "ENPPI", "القناة": "Olympic El Qanah FC", "المصري": "Al Masry",
    "المقاولون العرب": "Al Mokawloon Al Arab", "بترول أسيوط": "Asyut Petroleum", "بيراميدز": "Pyramids FC",
    "زد": "ZED FC", "سموحة": "Smouha SC", "سيراميكا كليوباترا": "Ceramica Cleopatra",
    "غزل المحلة": "Ghazl El Mehalla", "مودرن سبورت": "Modern Sport FC", "وادي دجلة": "Wadi Degla FC",
}


def _season_of(d) -> str:
    y = d.year if d.month >= 7 else d.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _result(h, a) -> str:
    return "H" if h > a else "A" if a > h else "D"


def kooora_matches() -> pd.DataFrame:
    """Current Egyptian season from Kooora: results and upcoming fixtures, English team names."""
    k = pd.read_csv(EGY_KOOORA, parse_dates=["date"])
    unknown = (set(k.home) | set(k.away)) - set(KOOORA_NAMES)
    if unknown:
        raise SystemExit(f"Add these Kooora team names to KOOORA_NAMES: {sorted(unknown)}")
    k["home"], k["away"] = k.home.map(KOOORA_NAMES), k.away.map(KOOORA_NAMES)
    k["date"] = k.date.dt.date
    k["season"] = k.date.map(_season_of)
    return k


def fetch_egypt_recent() -> pd.DataFrame:
    """2009/10 -> today: history from fetch_egypt.py, current season from Kooora (more complete and up to date)."""
    raw = pd.read_csv(EGY_RECENT, parse_dates=["date"])
    first_year = raw.season_slug.str.extract(r"(\d{4})-")[0].astype(int)
    df = raw.assign(date=raw.date.dt.date, season=first_year.map(lambda y: f"{y % 100:02d}{(y + 1) % 100:02d}"))
    if EGY_KOOORA.exists():
        k = kooora_matches()
        played = k[k.status == "RESULT"]
        df = pd.concat([df[df.date < played.date.min()], played], ignore_index=True)
    df["result"] = [_result(h, a) for h, a in zip(df.home_goals, df.away_goals)]
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
    # full sort key: same-day matches otherwise come out in a different order each run (Kooora page order),
    # which changed match_id and made the daily job commit identical data
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "league", "home", "away"]).reset_index(drop=True)
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
    epl_now = df[(df.league == "EPL") & (df.season == EPL_SEASONS[-1])]
    fixtures = [fetch_epl_fixtures(epl_now)]  # upcoming fixtures, so the app can analyse the next matches
    if EGY_KOOORA.exists():
        k = kooora_matches()
        fixtures.append(k[k.status == "FIXTURE"].assign(league="EGY")[["league", "season", "date", "home", "away"]])
    fixtures = pd.concat(fixtures, ignore_index=True).sort_values(["league", "date", "home", "away"])
    fixtures.to_sql("fixtures", eng, if_exists="replace", index=False)
    for league, part in fixtures.groupby("league"):
        print(f"{league} fixtures: {len(part)} upcoming")
    for league, part in df.groupby("league"):
        print(f"{league}: {len(part)} matches ({part.date.min()} -> {part.date.max()})")
    print(f"Saved to {DATABASE_URL}")


def content_hash() -> str:
    """Hash of the rows (not the file bytes, which change on every rebuild), so the daily job only commits real changes."""
    import hashlib
    h = hashlib.sha256()
    with engine().connect() as c:
        for table, order in (("matches", "league, date, home, away"), ("fixtures", "league, date, home, away")):
            try:
                for row in c.execute(text(f"SELECT * FROM {table} ORDER BY {order}")):
                    h.update(repr(tuple(row)).encode())
            except Exception:  # table missing on a fresh database
                h.update(table.encode())
    return h.hexdigest()


if __name__ == "__main__":
    run()
