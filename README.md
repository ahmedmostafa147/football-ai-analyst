# Football AI Match Analyst + Performance Dashboard

Pre-match analysis system: real match data → SQL → retrieval (SQL + RAG) → prediction model → LLM analyst → fact-checking agent → bilingual (English / Arabic) report, with a measurable evaluation.

**Leagues:** English Premier League (8,060 matches, 2005 → Oct 2026) and Egyptian Premier League (4,708 matches, 2009 → Oct 2026).

**Performance dashboard:** current-season team analysis, league overview (attack vs defence) and strengths & weaknesses for every team in both leagues.

**AI writer:** Gemini (`GEMINI_API_KEY`) or Claude (`ANTHROPIC_API_KEY`); without a key an offline template writes the report. Reports are cached and capped at 30 AI calls per day on the public site.

| Match Analyst | Performance Dashboard |
|---|---|
| ![Match analyst](docs/match_analyst.jpg) | ![Performance dashboard](docs/dashboard.jpg) |

```
football-data.co.uk + footballdatabase ─► pipeline.py ─► PostgreSQL / SQLite
                                                  │
                                  retrieval.py ◄──┘  SQL facts (table, form, home/away, head-to-head, xG)
                                                     + RAG over match documents (TF-IDF)
                                                     + team-strength prediction model
                                                  │
                                   agents.py ─────┤  AnalysisAgent (Claude) ⇄ ValidationAgent (fact checker, retry loop)
                                                  │
                           Match_Analyst.py / main.py ┘  web page / CLI → reports/*.md + *.json
                                 evaluate.py         backtest of all models + hallucination tests
```

## Run

```bash
pip install -r requirements.txt
python fetch_egypt.py                                 # Egyptian results 2009/10 -> today (~5 min, polite delays)
python pipeline.py                                    # load both leagues into SQL
streamlit run Match_Analyst.py                        # web app: match analyst + performance dashboard
python main.py "Arsenal" "Liverpool" --date 2026-11-01
python main.py "El Zamalek" "Al Ahly" --league EGY --season 2627 --date 2026-10-30
python evaluate.py
```

- **LLM mode:** set `ANTHROPIC_API_KEY` (model via `ANALYST_MODEL`, default `claude-sonnet-5-5`). Without a key, a template writer is used so the whole pipeline still runs.
- **PostgreSQL:** `DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/football` (install `psycopg2-binary`). Default is SQLite.

## Prediction model

| Version | Idea |
|---|---|
| v1 | Average goals for/against over the last year, shrunk towards the league average |
| **v2 (used)** | Dixon-Coles style Poisson regression: `log(goals) = home advantage + attack[team] − defence[opponent]`, fitted on the last 4 years with time decay (half-life 240 days) |
| v2 + H2H | v2 blended with the average goals in recent head-to-head meetings |

## Evaluation (no data leakage: refit every week on matches played before)

**EPL, seasons 2024/25 + 2025/26 (760 matches)**

| Model | Accuracy | Brier ↓ | Log loss ↓ |
|---|---|---|---|
| League base rate | 0.417 | 0.657 | 1.083 |
| v1 | 0.496 | 0.607 | 1.013 |
| **v2** | **0.504** | **0.602** | **1.005** |
| v2 + H2H (0.15) | 0.504 | 0.603 | 1.007 |
| v2 + H2H (0.3) | 0.500 | 0.605 | 1.011 |

**Egyptian league, seasons 2016/17 + 2017/18 (612 matches, older Kaggle data)**

| Model | Accuracy | Brier ↓ | Log loss ↓ |
|---|---|---|---|
| League base rate | 0.379 | 0.665 | 1.096 |
| v1 | 0.493 | 0.605 | 1.007 |
| **v2** | **0.500** | **0.602** | **1.001** |
| v2 + H2H (0.15) | 0.497 | 0.601 | 0.999 |
| v2 + H2H (0.3) | 0.503 | 0.601 | 0.999 |

**Egyptian league, seasons 2024/25 + 2025/26 (546 matches)**

| Model | Accuracy | Brier ↓ | Log loss ↓ |
|---|---|---|---|
| League base rate | 0.341 | 0.667 | 1.099 |
| v1 | 0.436 | 0.622 | 1.032 |
| **v2** | 0.416 | **0.621** | **1.030** |

The recent Egyptian league is harder to predict (many draws, tighter league), but both models still beat the base rate by a wide margin.

**Finding:** v2 beats v1 in both leagues. Head-to-head history adds almost nothing once current team strength is known (slightly worse in England, slightly better in Egypt), so it is shown in the report but not used in the prediction.

**Can machine learning do better?** (`python improve.py` — Elo, rolling goals, shots on target and points + v2 probabilities; trained on 2018/19–2023/24, tested on 2024/25–2025/26, 760 matches)

| Model | Accuracy | Brier ↓ | Log loss ↓ | Accuracy when model ≥ 60% sure (share of matches) |
|---|---|---|---|---|
| **v2** | 0.504 | **0.602** | **1.005** | **0.662** (26%) |
| Logistic regression, Elo only | 0.511 | 0.605 | 1.008 | 0.623 (30%) |
| Logistic regression, all features | 0.495 | 0.604 | 1.009 | 0.646 (30%) |
| Gradient boosting, all features | 0.496 | 0.627 | 1.045 | 0.599 (31%) |

No. Extra features and ML models do not beat v2 on probability quality. Around 50% is close to the practical ceiling for home/draw/away (about a quarter of matches end in draws, which no model predicts well). When v2 is confident (≥ 60%), it is right 66% of the time.

The project uses match and performance data only — no betting odds.

Fact checker: clean report passes; injected fake stat, fake score, fake probability and missing Arabic section are all caught.

## How hallucinations are prevented

1. The analyst only sees a JSON context built from SQL + retrieved documents.
2. `ValidationAgent` checks every number, score, team name and section in the report against that context.
3. On failure, the issues go back to the analyst as feedback (up to 3 attempts).

## Limitations / next steps

- Egyptian data has results only (no shots/xG); without `fetch_egypt.py` the pipeline falls back to the Kaggle 2010–2018 dataset.
- Team-level data only — current-season player stats need an API (e.g. API-Football) and are the next step.
- TF-IDF retrieval → swap for embeddings + pgvector.
