"""Retrieval layer: structured facts via SQL + relevant match documents via RAG (TF-IDF) + prediction model."""
import math
from datetime import date

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

from pipeline import engine

FORM_SQL = """
SELECT date, home, away, home_goals, away_goals, home_xg, away_xg
FROM matches
WHERE (home = :team OR away = :team) AND date < :before
ORDER BY date DESC LIMIT :n
"""

SPLIT_SQL = """
SELECT COUNT(*) AS played,
       AVG(CASE WHEN home = :team THEN home_goals ELSE away_goals END) AS scored,
       AVG(CASE WHEN home = :team THEN away_goals ELSE home_goals END) AS conceded,
       AVG(CASE WHEN home = :team THEN home_sot ELSE away_sot END) AS shots_on_target,
       AVG(CASE WHEN home = :team THEN home_xg ELSE away_xg END) AS xg_for,
       AVG(CASE WHEN home = :team THEN away_xg ELSE home_xg END) AS xg_against
FROM matches
WHERE {venue} = :team AND date < :before AND league = :league AND season = :season
"""

H2H_SQL = """
SELECT date, home, away, home_goals, away_goals FROM matches
WHERE ((home = :a AND away = :b) OR (home = :b AND away = :a)) AND date < :before
ORDER BY date DESC
"""


def _rows(sql: str, **params) -> list[dict]:
    with engine().connect() as c:
        return [dict(r._mapping) for r in c.execute(text(sql), params)]


def _matches(league: str, before: date, since: date | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM matches WHERE league = :l AND date < :d" + (" AND date >= :s" if since else "")
    with engine().connect() as c:
        df = pd.read_sql(text(sql), c, params={"l": league, "d": str(before), "s": str(since)})
    df["date"] = pd.to_datetime(df["date"])
    return df


def seasons(league: str) -> list[str]:
    rows = _rows("SELECT DISTINCT season, MIN(date) AS start FROM matches WHERE league = :l GROUP BY season ORDER BY start", l=league)
    return [r["season"] for r in rows]


def data_version(league: str) -> str:
    """Changes whenever the league's data is refreshed; pass it to cached functions so they never serve old numbers."""
    row = _rows("SELECT COUNT(*) AS n, MAX(date) AS last FROM matches WHERE league = :l", l=league)[0]
    return f"{row['n']}:{row['last']}"


def teams(league: str, season: str) -> list[str]:
    rows = _rows("SELECT DISTINCT home FROM matches WHERE league = :l AND season = :s ORDER BY home", l=league, s=season)
    return [r["home"] for r in rows]


def _points(row: dict, team: str) -> int:
    gf, ga = (row["home_goals"], row["away_goals"]) if row["home"] == team else (row["away_goals"], row["home_goals"])
    return 3 if gf > ga else 1 if gf == ga else 0


def form(team: str, before: date, n: int = 5) -> dict:
    rows = _rows(FORM_SQL, team=team, before=str(before), n=n)
    letters = "".join("WDL"[[3, 1, 0].index(_points(r, team))] for r in rows)
    games = [f"{r['date']} {r['home']} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away']}" for r in rows]
    return {"last_results": letters, "points": sum(_points(r, team) for r in rows), "games": games}


def split(team: str, venue: str, before: date, league: str, season: str) -> dict:
    row = _rows(SPLIT_SQL.format(venue=venue), team=team, before=str(before), league=league, season=season)[0]
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}


def head_to_head(a: str, b: str, before: date, show: int = 6) -> dict:
    rows = _rows(H2H_SQL, a=a, b=b, before=str(before))
    wins_a = sum(_points(r, a) == 3 for r in rows)
    wins_b = sum(_points(r, b) == 3 for r in rows)
    goals_a = sum(r["home_goals"] if r["home"] == a else r["away_goals"] for r in rows)
    goals_b = sum(r["home_goals"] if r["home"] == b else r["away_goals"] for r in rows)
    return {
        "meetings": len(rows), f"{a}_wins": wins_a, "draws": len(rows) - wins_a - wins_b, f"{b}_wins": wins_b,
        f"{a}_goals": int(goals_a), f"{b}_goals": int(goals_b),
        "recent": [f"{r['date']} {r['home']} {int(r['home_goals'])}-{int(r['away_goals'])} {r['away']}" for r in rows[:show]],
    }


def league_table(before: date, league: str, season: str) -> pd.DataFrame:
    df = _matches(league, before)
    df = df[df.season == season]
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({"team": df[side], "gf": df[f"{side}_goals"], "ga": df[f"{other}_goals"]})
        part["pts"] = (part.gf > part.ga) * 3 + (part.gf == part.ga) * 1
        parts.append(part)
    t = pd.concat(parts).groupby("team").agg(played=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"))
    t["gd"] = t.gf - t.ga
    t = t.sort_values(["pts", "gd", "gf"], ascending=False)
    t["position"] = range(1, len(t) + 1)
    return t.astype(int)


def _outcomes(lam_h: float, lam_a: float) -> dict:
    pmf = lambda lam, k: math.exp(-lam) * lam ** k / math.factorial(k)
    p = {"home": 0.0, "draw": 0.0, "away": 0.0}
    best, best_p = (0, 0), 0.0
    for i in range(10):
        for j in range(10):
            pij = pmf(lam_h, i) * pmf(lam_a, j)
            p["home" if i > j else "draw" if i == j else "away"] += pij
            if pij > best_p:
                best, best_p = (i, j), pij
    return {
        "expected_goals_home": round(lam_h, 2), "expected_goals_away": round(lam_a, 2),
        "p_home_win": round(p["home"], 3), "p_draw": round(p["draw"], 3), "p_away_win": round(p["away"], 3),
        "most_likely_score": f"{best[0]}-{best[1]}",
    }


def simple_poisson(home: str, away: str, before: date, league: str,
                   window_days: int = 365, prior_games: float = 8.0) -> dict:
    """v1 baseline: average goals for/against over the last year, shrunk towards the league average."""
    df = _matches(league, before, since=pd.Timestamp(before).date() - pd.Timedelta(days=window_days))
    avg_h, avg_a = df.home_goals.mean(), df.away_goals.mean()
    avg = (avg_h + avg_a) / 2

    def strength(team):
        h, a = df[df.home == team], df[df.away == team]
        games = len(h) + len(a)
        attack = (h.home_goals.sum() + a.away_goals.sum() + prior_games * avg) / ((games + prior_games) * avg)
        defence = (h.away_goals.sum() + a.home_goals.sum() + prior_games * avg) / ((games + prior_games) * avg)
        return attack, defence

    (h_att, h_def), (a_att, a_def) = strength(home), strength(away)
    return _outcomes(avg_h * h_att * a_def, avg_a * a_att * h_def)


class TeamStrengthModel:
    """v2: Dixon-Coles style Poisson regression.

    log(goals) = home_advantage + attack[team] - defence[opponent], fitted on the last few seasons
    with exponential time decay (recent matches count more), plus an optional head-to-head adjustment.
    """

    def __init__(self, league: str, before: date, half_life_days: float = 240,
                 window_years: float = 4, alpha: float = 0.002):
        df = _matches(league, before, since=pd.Timestamp(before).date() - pd.Timedelta(days=int(365 * window_years)))
        self.before = before
        self.teams = sorted(set(df.home) | set(df.away))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        X = np.zeros((2 * len(df), 2 * n + 1))
        y = np.concatenate([df.home_goals.values, df.away_goals.values])
        for k, (h, a) in enumerate(zip(df.home, df.away)):
            X[k, idx[h]] = 1; X[k, n + idx[a]] = 1; X[k, 2 * n] = 1           # home team scoring
            X[len(df) + k, idx[a]] = 1; X[len(df) + k, n + idx[h]] = 1          # away team scoring
        age = (pd.Timestamp(before) - df.date).dt.days.values
        weight = np.tile(0.5 ** (age / half_life_days), 2)
        self.idx, self.n = idx, n
        self.model = PoissonRegressor(alpha=alpha, max_iter=500).fit(X, y, sample_weight=weight)

    def _lambda(self, attack: str, defence: str, home: int) -> float:
        x = np.zeros((1, 2 * self.n + 1))
        if attack in self.idx:
            x[0, self.idx[attack]] = 1
        if defence in self.idx:
            x[0, self.n + self.idx[defence]] = 1
        x[0, 2 * self.n] = home
        return float(self.model.predict(x)[0])

    def predict(self, home: str, away: str, h2h_weight: float = 0.0, h2h_years: int = 8) -> dict:
        lam_h, lam_a = self._lambda(home, away, 1), self._lambda(away, home, 0)
        if h2h_weight:
            since = pd.Timestamp(self.before) - pd.Timedelta(days=365 * h2h_years)
            games = [r for r in _rows(H2H_SQL, a=home, b=away, before=str(self.before)) if pd.Timestamp(r["date"]) >= since]
            if games:
                n = len(games)
                gh = sum(r["home_goals"] if r["home"] == home else r["away_goals"] for r in games) / n
                ga = sum(r["home_goals"] if r["home"] == away else r["away_goals"] for r in games) / n
                w = h2h_weight * n / (n + 4)  # trust head-to-head more when there are more meetings
                lam_h = lam_h ** (1 - w) * max(gh, 0.2) ** w
                lam_a = lam_a ** (1 - w) * max(ga, 0.2) ** w
        return _outcomes(lam_h, lam_a)


class MatchRAG:
    """TF-IDF retriever over match documents (swap for embeddings + pgvector in production)."""

    def __init__(self, league: str, before: date):
        with engine().connect() as c:
            self.docs = pd.read_sql(text("SELECT * FROM documents WHERE league = :l AND date < :d"), c,
                                    params={"l": league, "d": str(before)})
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.docs.content)

    def search(self, query: str, k: int = 6) -> list[str]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [self.docs.content.iloc[i] for i in top if scores[i] > 0]


def build_context(home: str, away: str, kickoff: date, league: str, season: str) -> dict:
    table = league_table(kickoff, league, season)
    pos = lambda t: table.loc[t].to_dict() if t in table.index else None
    return {
        "match": {"home": home, "away": away, "date": str(kickoff), "league": league, "season": season},
        "table": {home: pos(home), away: pos(away)},
        "form": {home: form(home, kickoff), away: form(away, kickoff)},
        "home_record": split(home, "home", kickoff, league, season),
        "away_record": split(away, "away", kickoff, league, season),
        "head_to_head": head_to_head(home, away, kickoff),
        "model": TeamStrengthModel(league, kickoff).predict(home, away),
        "retrieved_documents": MatchRAG(league, kickoff).search(f"{home} {away} {home} vs {away}"),
    }
