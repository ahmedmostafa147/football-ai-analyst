# Handoff — Football AI Analyst

Last updated: 2026-10-31. Read `AGENTS.md` first for rules and architecture.

## Why this exists

The owner (a Flutter developer, ~4.5 years) is moving into **AI Engineer / football data & performance analysis**
roles in the Gulf, Europe or remote, and needs one or two strong portfolio projects he can explain in an interview.
Plan agreed with him: finish this project → he learns it step by step → update CV → apply to AI Engineer roles
(Gulf has the most openings) → later move into a football company or club.

## Live

- App: https://football-ai-analyst-v1.streamlit.app (Streamlit Community Cloud, owner's account, deploys on push to `main`)
- Repo: https://github.com/ahmedmostafa147/football-ai-analyst (public)
- Gemini key is set in Streamlit Secrets and verified: reports show "written by Gemini" and pass the fact check.

## Done (verified in the browser)

- Data: EPL 8,060 matches (2005 → 25 Oct 2026) + Egyptian league 4,721 matches (2009 → 30 Oct 2026).
  Egyptian table matches Kooora on 30 Oct 2026: Pyramids 24, Al Ahly 20, Zamalek 19 (3rd).
- Upcoming fixtures for both leagues; the match page opens on the next real fixture
  (EGY: El Gounah vs El Zamalek 23 Nov; EPL: Liverpool vs Arsenal 1 Nov).
- Match Analyst: probability bar, team cards, head-to-head, Arabic + English report (headings + bullets), RAG tab.
- Performance Dashboard on the current season: team analysis, league overview (attack vs defence), strengths & weaknesses.
- AI writer: Gemini (with 503 retry + lite fallback) → Claude → offline template; 30 AI reports/day cap; 24h cache.
- Evaluation (in README): v2 model ~50% accuracy EPL, ~42% Egypt (recent seasons) vs 34–42% base rate;
  3 ML models and head-to-head weighting tested and rejected with numbers; fact checker catches 4/4 injected errors.

## Not done / next steps (in priority order)

1. **Owner understanding.** He said he doesn't understand the project yet. Teach it in 4 short lessons following the
   pipeline (data → retrieval → writer → checker), in Egyptian Arabic, with one interview question after each.
2. **Automatic data refresh.** Built 2026-11-03: daily GitHub Action (`.github/workflows/refresh-data.yml`).
   Check the Actions tab if the site looks stale; Kooora blocking GitHub's servers is the most likely failure.
3. **Player analysis for the current season.** No free, allowed source found. Plan: owner creates a free
   API-Football key; test whether the free plan includes season 2026 before building anything.
4. **Rotate the Gemini key.** It was pasted into a chat on 2026-10-31; the owner was advised to create a new one.
5. Engineering polish a senior reviewer will notice: unit tests (`pytest`) for pipeline/validator, move modules into a
   package folder, embeddings instead of TF-IDF for RAG.
6. Send the project to a senior for feedback — a ready-to-send Arabic message was drafted in the session
   (summary: what it does, how it works, what was measured, known gaps, ask for honest feedback).

## Known limitations

- Egyptian data has goals only (no shots/xG), so Egyptian dashboard metrics are goal-based.
- EPL future dates come from openfootball and show Saturday 15:00 until TV picks are known; football-data's
  `fixtures.csv` corrects the next round once it lists it.
- `TeamStrengthModel` ignores promoted teams' history from lower divisions (they start at league average).
