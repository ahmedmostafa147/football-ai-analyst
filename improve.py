"""Experiment: can machine learning on extra features beat the v2 model? Train on older seasons, test on the last two."""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, build

OUTCOMES = ["H", "D", "A"]
SETUPS = {"EPL": ("1819", ["2425", "2526"])}


def score(probs: np.ndarray, y: pd.Series) -> dict:
    actual = y.map(OUTCOMES.index).values
    picked = probs.argmax(1)
    confident = probs.max(1) >= 0.6
    onehot = np.eye(3)[actual]
    return {
        "accuracy": round((picked == actual).mean(), 3),
        "brier": round(((probs - onehot) ** 2).sum(1).mean(), 4),
        "log_loss": round(-np.log(np.clip(probs[np.arange(len(actual)), actual], 1e-9, 1)).mean(), 4),
        "confident_share": round(confident.mean(), 3),
        "confident_accuracy": round((picked[confident] == actual[confident]).mean(), 3) if confident.any() else None,
    }


def run(league: str, first: str, test_seasons: list[str]) -> pd.DataFrame:
    df = build(league, first)
    df[FEATURES] = df[FEATURES].fillna(0)
    test = df[df.season.isin(test_seasons)]
    train = df[df.date < test.date.min()]  # only seasons before the test period
    results = {"v2 (current)": score(test[["v2_home", "v2_draw", "v2_away"]].values, test.result)}
    models = {
        "logistic (elo only)": (["elo_diff"], LogisticRegression(max_iter=1000)),
        "logistic (all features)": (FEATURES, LogisticRegression(max_iter=1000)),
        "gradient boosting (all)": (FEATURES, HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200)),
    }
    for name, (cols, clf) in models.items():
        pipe = make_pipeline(StandardScaler(), clf).fit(train[cols], train.result)
        order = [list(pipe.classes_).index(o) for o in OUTCOMES]
        results[name] = score(pipe.predict_proba(test[cols])[:, order], test.result)
    print(f"\n== {league}: train {train.season.nunique()} seasons ({len(train)} matches), "
          f"test {', '.join(test_seasons)} ({len(test)} matches) ==")
    table = pd.DataFrame(results).T
    print(table.to_string())
    return table


if __name__ == "__main__":
    for league, (first, test_seasons) in SETUPS.items():
        run(league, first, test_seasons)
