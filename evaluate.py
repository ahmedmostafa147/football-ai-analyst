"""Evaluation: (1) backtest prediction models with no data leakage, (2) test the fact checker."""
import json
import math
from datetime import date

import pandas as pd

from agents import ValidationAgent, template_report
from retrieval import TeamStrengthModel, _matches, build_context, simple_poisson

OUTCOMES = ("H", "D", "A")
BACKTESTS = {"EPL": ["2425", "2526"], "EGY": ["2425", "2526"]}
H2H_WEIGHTS = (0.15, 0.3)


def _probs(m: dict) -> tuple:
    return m["p_home_win"], m["p_draw"], m["p_away_win"]


def backtest(league: str, test_seasons: list[str]) -> dict:
    everything = _matches(league, date(2100, 1, 1))
    test = everything[everything.season.isin(test_seasons)].sort_values("date")
    preds = {"base_rate": [], "v1_simple": [], "v2_strength": [], **{f"v2_h2h_{w}": [] for w in H2H_WEIGHTS}}
    results = []
    # refit once per week: every prediction only uses matches played before that week
    for _, week in test.groupby(test.date.dt.to_period("W")):
        kickoff = week.date.min().date()
        history = everything[everything.date < week.date.min()]
        rates = history.result.value_counts(normalize=True)
        model = TeamStrengthModel(league, kickoff)
        for r in week.itertuples():
            results.append(r.result)
            preds["base_rate"].append(tuple(float(rates.get(o, 0)) for o in OUTCOMES))
            preds["v1_simple"].append(_probs(simple_poisson(r.home, r.away, kickoff, league)))
            preds["v2_strength"].append(_probs(model.predict(r.home, r.away)))
            for w in H2H_WEIGHTS:
                preds[f"v2_h2h_{w}"].append(_probs(model.predict(r.home, r.away, h2h_weight=w)))

    def score(probs_list):
        acc = brier = logloss = 0.0
        for probs, res in zip(probs_list, results):
            actual = OUTCOMES.index(res)
            acc += probs.index(max(probs)) == actual
            brier += sum((p - (i == actual)) ** 2 for i, p in enumerate(probs))
            logloss -= math.log(max(probs[actual], 1e-9))
        n = len(results)
        return {"accuracy": round(acc / n, 3), "brier": round(brier / n, 4), "log_loss": round(logloss / n, 4)}

    return {"league": league, "seasons": test_seasons, "matches": len(results),
            **{name: score(p) for name, p in preds.items()}}


def fact_checker_test() -> dict:
    """Inject hallucinations into a grounded report and check the validator catches each one."""
    ctx = build_context("Arsenal", "Liverpool", date(2026, 11, 1), "EPL", "2627")
    clean = template_report(ctx)
    checker = ValidationAgent()
    injections = {
        "fake_stat": clean.replace("## English", "## English\nArsenal have scored 47 goals this season."),
        "fake_score": clean.replace("## English", "## English\nLast season Arsenal won 9-7 at Anfield."),
        "fake_probability": clean.replace("## English", "## English\nArsenal win chance is 81.2%."),
        "missing_arabic": clean.split("## العربية")[0],
    }
    caught = {name: not checker.run(report, ctx)["passed"] for name, report in injections.items()}
    return {"clean_report_passes": checker.run(clean, ctx)["passed"], "hallucinations_caught": caught}


if __name__ == "__main__":
    for league, test_seasons in BACKTESTS.items():
        res = backtest(league, test_seasons)
        print(f"\n== {res['league']} backtest: seasons {', '.join(res['seasons'])}, {res['matches']} matches ==")
        table = pd.DataFrame({k: v for k, v in res.items() if isinstance(v, dict)}).T
        print(table.to_string())
    print("\n== Fact-checker test ==")
    print(json.dumps(fact_checker_test(), indent=2))
