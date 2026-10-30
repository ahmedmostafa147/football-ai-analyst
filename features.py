"""Pre-match features for every match, computed strictly from earlier matches (no leakage)."""
from collections import defaultdict, deque
from datetime import date

import numpy as np
import pandas as pd

from retrieval import TeamStrengthModel, _matches

FEATURES = ["elo_diff", "v2_home", "v2_draw", "v2_away", "gf_diff", "ga_diff", "sot_diff", "pts_diff"]


def elo_and_form(df: pd.DataFrame, k: float = 20, home_adv: float = 60, window: int = 10) -> pd.DataFrame:
    """Elo rating difference + rolling form (goals, shots on target, points) before each match."""
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(lambda: deque(maxlen=window))  # (gf, ga, sot_for, pts)
    out = []
    for r in df.itertuples():
        def avg(team, i):
            h = hist[team]
            vals = [x[i] for x in h if not pd.isna(x[i])]
            return np.mean(vals) if vals else np.nan

        out.append({
            "elo_diff": elo[r.home] + home_adv - elo[r.away],
            "gf_diff": avg(r.home, 0) - avg(r.away, 0),
            "ga_diff": avg(r.home, 1) - avg(r.away, 1),
            "sot_diff": avg(r.home, 2) - avg(r.away, 2),
            "pts_diff": avg(r.home, 3) - avg(r.away, 3),
        })
        expected = 1 / (1 + 10 ** (-(elo[r.home] + home_adv - elo[r.away]) / 400))
        actual = {"H": 1.0, "D": 0.5, "A": 0.0}[r.result]
        margin = np.log1p(abs(r.home_goals - r.away_goals)) + 1  # bigger wins move ratings more
        elo[r.home] += k * margin * (actual - expected)
        elo[r.away] -= k * margin * (actual - expected)
        pts_h = 3 if r.result == "H" else 1 if r.result == "D" else 0
        pts_a = 3 if r.result == "A" else 1 if r.result == "D" else 0
        hist[r.home].append((r.home_goals, r.away_goals, r.home_sot, pts_h))
        hist[r.away].append((r.away_goals, r.home_goals, r.away_sot, pts_a))
    return pd.DataFrame(out, index=df.index)


def build(league: str, from_season: str) -> pd.DataFrame:
    """All matches of a league with features; v2 probabilities only from `from_season` on (refit weekly)."""
    df = _matches(league, date(2100, 1, 1)).sort_values("date").reset_index(drop=True)
    for col in ("home_goals", "away_goals", "home_sot", "away_sot"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.join(elo_and_form(df))
    start = df[df.season == from_season].date.min()
    probs = {}
    for _, week in df[df.date >= start].groupby(df.date.dt.to_period("W")):
        model = TeamStrengthModel(league, week.date.min().date())
        for i, r in week.iterrows():
            m = model.predict(r.home, r.away)
            probs[i] = (m["p_home_win"], m["p_draw"], m["p_away_win"])
    p = pd.DataFrame.from_dict(probs, orient="index", columns=["v2_home", "v2_draw", "v2_away"])
    return df.join(p).loc[p.index]
