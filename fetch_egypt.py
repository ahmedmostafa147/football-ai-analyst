"""Download Egyptian Premier League results (2009/10 -> today) from footballdatabase.com into data/egypt_results.csv.

Run it yourself:  python fetch_egypt.py
It waits between requests so it doesn't overload the site.
"""
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://footballdatabase.com"
OUT = Path(__file__).with_name("data") / "egypt_results.csv"
SEASON_DIR = OUT.parent / "egypt_seasons"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal football analysis project)"}
DELAY_SECONDS = 4

MATCH = re.compile(
    r'class="club-gamelist-match-clubs"[^>]*>\s*<a[^>]*title="(?P<home>[^"]+)"'
    r'.*?class="club-gamelist-match-score[^"]*">\s*(?P<hg>\d+)\s*-\s*(?P<ag>\d+)\s*<'
    r'.*?class="club-gamelist-match-clubs"[^>]*>\s*<a[^>]*title="(?P<away>[^"]+)"',
    re.S,
)
DAY = re.compile(r"<h4>(\d{1,2} \w+ \d{4})</h4>")


def parse_results(html: str) -> list[dict]:
    """Each <h4>date</h4> is followed by that day's matches."""
    rows = []
    days = list(DAY.finditer(html))
    for i, day in enumerate(days):
        end = days[i + 1].start() if i + 1 < len(days) else len(html)
        played = datetime.strptime(day.group(1), "%d %B %Y").date()
        for m in MATCH.finditer(html, day.end(), end):
            rows.append({"date": played, "home": m["home"], "away": m["away"],
                         "home_goals": int(m["hg"]), "away_goals": int(m["ag"])})
    return rows


def get(url: str, attempts: int = 5) -> str:
    """GET with backoff: the site sometimes drops connections when asked too quickly."""
    for attempt in range(attempts):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            wait = 15 * (attempt + 1)
            print(f"  {e.__class__.__name__}, retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Giving up on {url} - run the script again later, finished seasons are kept.")


def season_slugs() -> list[str]:
    html = get(f"{BASE}/league-scores-tables/egypt-premier-league-2024-2025")
    return sorted(set(re.findall(r'value="/league-scores-tables/(egypt-premier-league-[\d-]+)"', html)))


def fetch_season(slug: str) -> list[dict]:
    rows, page = [], 1
    while True:
        html = get(f"{BASE}/league-scores/{slug}/{page}")
        found = parse_results(html)
        if not found:
            break
        rows += found
        if f"league-scores/{slug}/{page + 1}" not in html:
            break
        page += 1
        time.sleep(DELAY_SECONDS)
    for r in rows:
        r["season_slug"] = slug
    return rows


def main():
    SEASON_DIR.mkdir(parents=True, exist_ok=True)
    slugs = season_slugs()
    for slug in slugs:
        cached = SEASON_DIR / f"{slug}.csv"
        if cached.exists() and slug != slugs[-1]:  # finished seasons never change; always refresh the current one
            print(f"{slug}: cached")
            continue
        rows = fetch_season(slug)
        print(f"{slug}: {len(rows)} matches")
        if rows:
            pd.DataFrame(rows).to_csv(cached, index=False)
        time.sleep(DELAY_SECONDS)
    parts = [pd.read_csv(f) for f in sorted(SEASON_DIR.glob("*.csv"))]
    if not parts:
        sys.exit("No matches downloaded - the site layout may have changed.")
    df = pd.concat(parts).drop_duplicates(["date", "home", "away"]).sort_values("date")
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} matches to {OUT}")


if __name__ == "__main__":
    main()
