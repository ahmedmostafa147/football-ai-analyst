"""Download the current Egyptian Premier League season (results + upcoming fixtures) from Kooora.

Kooora is updated within hours of each match, so it is the source for the current season; older seasons
still come from fetch_egypt.py. Each team page lists that team's matches, so we visit all 20 teams and merge.

Run:  python fetch_kooora.py
"""
import json
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.kooora.com"
LEAGUE_ID = "8k1xcsyvxapl4jlsluh3eomre"  # الدوري المصري الممتاز
ANY_TEAM_ID = "2c1lvho9eqtqfqf4t3cv0f237"  # Zamalek: its table page lists every team in the league
OUT = Path(__file__).with_name("data") / "kooora_egypt.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal football analysis project)"}
DELAY_SECONDS = 2


def page_data(path: str) -> dict:
    url = BASE + urllib.parse.quote(path)
    for attempt in range(4):
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            raw = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.S).group(1)
            return json.loads(raw)["props"]["pageProps"]["data"]
        except (requests.RequestException, AttributeError) as e:
            print(f"  {e.__class__.__name__}, retrying")
            time.sleep(10 * (attempt + 1))
    raise SystemExit(f"Could not read {url}")


def league_teams() -> list[dict]:
    data = page_data(f"/كرة-القدم/فريق/x/جدول/{ANY_TEAM_ID}")
    table = next(s for s in data["standings"] if s["competition"]["id"] == LEAGUE_ID)
    print(f"Table last updated {table['table']['lastUpdatedAt']}")
    return [r["team"] for r in table["table"]["rankings"]]


def team_matches(team_id: str) -> list[dict]:
    rows = []
    for m in page_data(f"/كرة-القدم/فريق/x/مباريات/{team_id}")["matches"]:
        if m["competition"]["id"] != LEAGUE_ID or m["status"] not in ("RESULT", "FIXTURE"):
            continue
        score = m.get("score") or {}
        rows.append({
            "match_id": m["id"], "kickoff_utc": m["startDate"], "status": m["status"],
            "home": m["teamA"]["name"], "away": m["teamB"]["name"],
            "home_goals": score.get("teamA"), "away_goals": score.get("teamB"),
            "home_red": (m.get("redCards") or {}).get("teamA"), "away_red": (m.get("redCards") or {}).get("teamB"),
        })
    return rows


def main():
    rows = []
    for team in league_teams():
        found = team_matches(team["id"])
        print(f"{team['name']}: {len(found)} league matches")
        rows += found
        time.sleep(DELAY_SECONDS)
    df = pd.DataFrame(rows).drop_duplicates("match_id")
    df["date"] = pd.to_datetime(df.kickoff_utc).dt.tz_convert("Africa/Cairo").dt.date
    df = df.sort_values("kickoff_utc")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    played = df[df.status == "RESULT"]
    print(f"Saved {len(played)} results ({played.date.min()} -> {played.date.max()}) "
          f"and {(df.status == 'FIXTURE').sum()} upcoming fixtures to {OUT}")


if __name__ == "__main__":
    main()
