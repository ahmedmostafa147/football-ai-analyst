"""Player & team performance data from StatsBomb open data (free event data: every pass, shot, tackle).

Downloads each match's events once, aggregates them per team and per player, and saves to SQL:
  sb_matches       one row per match
  sb_team_match    team stats per match (goals, shots, xG, possession, passes...)
  sb_player_match  player stats per match (minutes, shots, key passes, dribbles, tackles...)
Raw event files are cached in .cache/statsbomb/ so re-runs are fast.

Run:  python statsbomb.py
"""
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from pipeline import engine

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = Path(__file__).with_name(".cache") / "statsbomb"
COMPETITIONS = {  # (competition_id, season_id): label
    (1267, 107): "AFCON 2023",
    (43, 106): "World Cup 2022",
    (55, 282): "Euro 2024",
}
ON_TARGET = {"Goal", "Saved", "Saved to Post"}
WON = {"Won", "Success In Play", "Success Out"}


def _json(path: str):
    local = CACHE / path
    if local.exists():
        return json.loads(local.read_text(encoding="utf-8"))
    resp = requests.get(f"{RAW}/{path}", timeout=60)
    resp.raise_for_status()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(resp.text, encoding="utf-8")
    return resp.json()


def _minutes(events: list[dict]) -> dict:
    """Minutes played per player from Starting XI + substitutions."""
    end = max(e["minute"] for e in events if e["period"] <= 4) + 1
    on, off, position = {}, {}, {}
    for e in events:
        if e["type"]["name"] == "Starting XI":
            for p in e["tactics"]["lineup"]:
                on[p["player"]["name"]] = 0
                position[p["player"]["name"]] = p["position"]["name"]
        elif e["type"]["name"] == "Substitution" and e["period"] <= 4:
            off[e["player"]["name"]] = e["minute"]
            on[e["substitution"]["replacement"]["name"]] = e["minute"]
    return {p: (off.get(p, end) - start, position.get(p)) for p, start in on.items()}


def aggregate(match: dict, competition: str):
    events = [e for e in _json(f"events/{match['match_id']}.json") if e["period"] <= 4]  # no penalty shootouts
    teams = (match["home_team"]["home_team_name"], match["away_team"]["away_team_name"])
    team = {t: Counter() for t in teams}
    player = defaultdict(Counter)
    player_team, player_pos = {}, defaultdict(Counter)
    possession = Counter()

    for e in events:
        t, kind = e["team"]["name"], e["type"]["name"]
        possession[e["possession_team"]["name"]] += e.get("duration") or 0
        name = e.get("player", {}).get("name")
        if name:
            player_team[name] = t
            if e.get("position"):
                player_pos[name][e["position"]["name"]] += 1
        p = player[name] if name else Counter()
        if kind == "Shot":
            xg, outcome = e["shot"].get("statsbomb_xg", 0), e["shot"]["outcome"]["name"]
            for c in (team[t], p):
                c["shots"] += 1; c["xg"] += xg
                c["shots_on_target"] += outcome in ON_TARGET
                c["goals"] += outcome == "Goal"
        elif kind == "Own Goal For":
            team[t]["goals"] += 1
        elif kind == "Pass":
            done = "outcome" not in e["pass"]
            for c in (team[t], p):
                c["passes"] += 1; c["passes_completed"] += done
                c["key_passes"] += bool(e["pass"].get("shot_assist") or e["pass"].get("goal_assist"))
                c["assists"] += bool(e["pass"].get("goal_assist"))
                c["crosses"] += bool(e["pass"].get("cross"))
        elif kind == "Dribble":
            for c in (team[t], p):
                c["dribbles"] += 1; c["dribbles_completed"] += e["dribble"]["outcome"]["name"] == "Complete"
        elif kind == "Duel" and e["duel"].get("type", {}).get("name") == "Tackle":
            for c in (team[t], p):
                c["tackles_won"] += e["duel"].get("outcome", {}).get("name") in WON
        elif kind == "Interception":
            for c in (team[t], p):
                c["interceptions"] += e["interception"]["outcome"]["name"] in WON
        elif kind == "Ball Recovery":
            for c in (team[t], p):
                c["recoveries"] += 1
        elif kind == "Foul Committed":
            team[t]["fouls"] += 1

    total_poss = sum(possession[t] for t in teams) or 1
    team_rows = []
    for t, opp in (teams, teams[::-1]):
        team_rows.append({
            "match_id": match["match_id"], "competition": competition, "team": t, "opponent": opp,
            **team[t], "goals_against": team[opp]["goals"], "xg_against": team[opp]["xg"],
            "shots_against": team[opp]["shots"], "possession": 100 * possession[t] / total_poss,
        })
    minutes = _minutes(events)
    player_rows = [{
        "match_id": match["match_id"], "competition": competition, "player": name,
        "team": player_team.get(name), "minutes": minutes.get(name, (0, None))[0],
        "position": minutes.get(name, (0, None))[1] or (player_pos[name].most_common(1)[0][0] if player_pos[name] else None),
        **stats,
    } for name, stats in player.items() if name]
    return team_rows, player_rows


def run():
    matches, team_rows, player_rows = [], [], []
    for (comp, season), label in COMPETITIONS.items():
        comp_matches = _json(f"matches/{comp}/{season}.json")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda m: aggregate(m, label), comp_matches))
        for m, (t, p) in zip(comp_matches, results):
            matches.append({
                "match_id": m["match_id"], "competition": label, "date": m["match_date"],
                "stage": m["competition_stage"]["name"],
                "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
                "home_score": m["home_score"], "away_score": m["away_score"],
            })
            team_rows += t
            player_rows += p
        print(f"{label}: {len(comp_matches)} matches")
    eng = engine()
    pd.DataFrame(matches).to_sql("sb_matches", eng, if_exists="replace", index=False)
    pd.DataFrame(team_rows).fillna(0).to_sql("sb_team_match", eng, if_exists="replace", index=False)
    pd.DataFrame(player_rows).fillna({"position": "Unknown"}).fillna(0).to_sql("sb_player_match", eng, if_exists="replace", index=False)
    print(f"Saved {len(team_rows)} team rows and {len(player_rows)} player rows")


if __name__ == "__main__":
    run()
