# Football AI Analyst — instructions for any AI agent

Read this file first, then `HANDOFF.md` (current status and next steps). `CLAUDE.md` points here.

## What this project is

A portfolio project for an AI Engineer / football data role. Two Streamlit pages, live at
https://football-ai-analyst-v1.streamlit.app (repo: github.com/ahmedmostafa147/football-ai-analyst, branch `main`).

1. **Match Analyst** (`Match_Analyst.py`): pick an upcoming fixture → SQL facts + RAG over past matches + Poisson
   team-strength model → LLM (Gemini or Claude) writes a pre-match report in English and Arabic → a validation agent
   checks every number in the report against the data and asks for a rewrite if one is invented.
2. **Performance Dashboard** (`pages/2_Performance_Dashboard.py`): current-season team analysis, league overview,
   strengths & weaknesses for the English and Egyptian Premier League.

## Hard rules (from the owner)

- **No betting.** Never load, show or compare against betting odds, and never suggest betting companies or features.
- **Current data only on screen.** The owner rejected showing old tournaments (2022–2024). Default to the current season.
- **Numbers must be right.** Before claiming a table, standings or result, cross-check with a live source (Kooora for
  Egypt). An earlier source silently missed matches and the app showed Zamalek 4th instead of 3rd.
- **Never write API keys into files or commits.** Keys live in Streamlit Cloud → app → Settings → Secrets
  (`GEMINI_API_KEY`, optional `ANTHROPIC_API_KEY`) and in local environment variables only.
- **Test in the browser before saying it works**, and say plainly what was not verified.
- The owner speaks Egyptian Arabic and wants short, concrete answers: what changed, what's left, one next step.

## Architecture (one line per file)

| File | Role |
|---|---|
| `pipeline.py` | Builds `football.db` (SQLite; Postgres via `DATABASE_URL`): tables `matches`, `documents` (RAG text), `fixtures` |
| `fetch_egypt.py` | Egyptian history 2009/10 → 2025/26 from footballdatabase.com → `data/egypt_seasons/*.csv` (cached per season) |
| `fetch_kooora.py` | Current Egyptian season results + upcoming fixtures from Kooora → `data/kooora_egypt.csv` |
| `retrieval.py` | SQL facts (table, form, home/away, head-to-head), `TeamStrengthModel` (v2), `MatchRAG` (TF-IDF), `build_context` |
| `agents.py` | `AnalysisAgent` (Gemini → Claude → offline template), `ValidationAgent` (fact checker), `orchestrate` (retry loop) |
| `report_template.py` | Offline report writer, same structure as the LLM prompt |
| `ui.py` | Theme, CSS, HTML components, Plotly template, markdown→HTML report renderer (RTL-safe) |
| `evaluate.py` / `improve.py` / `features.py` | Backtests (no leakage), ML experiments, fact-checker hallucination tests |
| `main.py` | CLI version of the match analyst |

Data sources: EPL results + fixtures from football-data.co.uk (`E0.csv`, `fixtures.csv`); Egyptian current season from
Kooora (Arabic names mapped to English in `pipeline.KOOORA_NAMES`); Egyptian history from footballdatabase.com.
Do **not** use Understat (robots.txt disallows all). FBref/worldfootball/soccerstats block automated requests.

## Refreshing data (manual for now)

```bash
python fetch_kooora.py     # current Egyptian season + fixtures (~1 min)
python pipeline.py         # rebuilds football.db (EPL is re-downloaded here)
git add football.db && git commit -m "Refresh data" && git push   # Streamlit Cloud redeploys from main
```
`football.db` is committed on purpose so the cloud app runs without downloading. `data/` is git-ignored.
If Kooora shows a new team name, `pipeline.py` stops and asks you to add it to `KOOORA_NAMES`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run Match_Analyst.py          # run from inside football_ai/ (theme lives in .streamlit/config.toml)
python evaluate.py                       # backtests + fact-checker tests; must stay green
```
Streamlit does not reload edited helper modules (`ui.py`, `agents.py`…): restart the server after editing them.

## Gotchas learned the hard way

- **Fact checker:** any number the template/LLM computes itself (e.g. a probability gap) fails validation. Only print
  numbers that exist in the context JSON.
- **Gemini:** `gemini-flash-latest` often returns 503 under load; `agents.py` retries then falls back to
  `gemini-flash-lite-latest`. `gemini-2.5-flash` is no longer available to new users.
- **Arabic rendering:** `ui.report_html(..., rtl=True)` wraps English names/scores in isolated LTR spans; keep it.
- **Report cache:** `analyse()` in `Match_Analyst.py` is cached 24h; `use_ai` is part of the key. Daily AI cap = 30.
- **Windows + heredocs:** writing Python that contains `\n` or `\1` through a bash heredoc corrupted files several
  times. Edit files with an editor tool instead of shell heredocs.
- **Team names:** EPL uses football-data names (`Man United`, `Nott'm Forest`); Egypt uses `El Zamalek`, `Al Ahly`, …
