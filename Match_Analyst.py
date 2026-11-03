"""Web page: streamlit run Match_Analyst.py"""
import os
import time
from datetime import date, timedelta

import streamlit as st
from sqlalchemy import text

import ui
from agents import orchestrate
from pipeline import LEAGUES, engine
from retrieval import build_context, data_version, seasons, teams

# default matchup per league (names differ between the Kaggle and footballdatabase sources)
DEFAULTS = {"EPL": (["Arsenal"], ["Liverpool"]),
            "EGY": (["El Zamalek", "Zamalek SC"], ["Al Ahly", "El Ahly Cairo"])}


def _pick(names: list[str], options: list[str], fallback: int) -> int:
    return next((options.index(n) for n in names if n in options), fallback)


MAX_REPORTS_PER_DAY = 30  # protects the API key on the public site; repeated fixtures are served from cache


@st.cache_resource
def _usage() -> dict:
    return {}


for _key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY"):  # Streamlit Cloud secrets -> environment for agents.py
    try:
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
    except Exception:  # no secrets file locally
        pass


def _ai_allowed() -> bool:
    usage, today = _usage(), str(date.today())
    return usage.get(today, 0) < MAX_REPORTS_PER_DAY


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def context(home: str, away: str, kickoff: date, league: str, season: str, version: str) -> dict:
    """SQL facts + RAG + model. version (data_version) is in the cache key, so a data refresh gives fresh facts."""
    return build_context(home, away, kickoff, league, season)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def report(home: str, away: str, kickoff: date, league: str, season: str, version: str, use_ai: bool) -> dict:
    """use_ai is part of the cache key, so adding an API key or hitting the daily cap gives a fresh report."""
    if use_ai:
        usage, today = _usage(), str(date.today())
        usage[today] = usage.get(today, 0) + 1
    return orchestrate(context(home, away, kickoff, league, season, version), use_ai=use_ai)


def upcoming(league: str, season: str, limit: int = 40) -> list[dict]:
    """Next fixtures (Egyptian league, from Kooora); empty when the league has no fixture list."""
    with engine().connect() as c:
        try:
            rows = c.execute(text("SELECT date, home, away FROM fixtures WHERE league = :l AND season = :s "
                                  "AND date >= :today ORDER BY date LIMIT :n"),
                             {"l": league, "s": season, "today": str(date.today()), "n": limit})
            return [dict(r._mapping) for r in rows]
        except Exception:  # no fixtures table yet
            return []


ui.setup("Match Analyst", "⚽")

with engine().connect() as c:
    counts = dict(c.execute(text("SELECT league, COUNT(*) FROM matches GROUP BY league")).all())
ui.hero("Pre-match intelligence", "Match Analyst",
        "Pick a fixture. The system pulls the facts with SQL, retrieves similar past matches, "
        "predicts the outcome and writes a fact-checked report in English and Arabic.",
        [f"{n:,} {LEAGUES[l]} matches" for l, n in counts.items()] + ["SQL · RAG · LLM · fact checking"])

with st.container(border=True):
    league = st.segmented_control("League", list(LEAGUES), format_func=LEAGUES.get, default="EGY") or "EGY"
    season = seasons(league)[-1]  # current season only (owner's rule: no old tournaments on screen)
    team_list = teams(league, season)
    with engine().connect() as c:
        last_played = c.execute(text("SELECT MAX(date) FROM matches WHERE league = :l AND season = :s"),
                                {"l": league, "s": season}).scalar()
    d_home, d_away = DEFAULTS[league]
    fixtures = upcoming(league, season)
    fixture = None
    if fixtures:
        favourite = next((i for i, f in enumerate(fixtures) if set(d_home) & {f["home"], f["away"]}), 0)
        # "pick teams" first so it is easy to find in a 40-match list; the page still opens on a real fixture
        options = ["✎ Pick any two teams yourself"] + [f"{ui.nice_date(f['date'])} · {f['home']} vs {f['away']}"
                                                       for f in fixtures]
        choice = st.selectbox(f"Upcoming fixture · season 20{season[:2]}/{season[2:]}", range(len(options)),
                              index=favourite + 1, format_func=options.__getitem__)
        fixture = fixtures[choice - 1] if choice else None
    if fixture:  # the fixture line already shows teams and date, so no extra fields (they pushed the button off-screen on phones)
        home, away, kickoff = fixture["home"], fixture["away"], date.fromisoformat(fixture["date"])
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        home = c1.selectbox("Home team", team_list, index=_pick(d_home, team_list, 0))
        away = c2.selectbox("Away team", team_list, index=_pick(d_away, team_list, 1))
        kickoff = c3.date_input("Match date", max(date.today(), date.fromisoformat(str(last_played)) + timedelta(days=1)))
    run = st.button("Analyse match", type="primary", width="stretch")

if run:
    if home == away:
        st.error("Pick two different teams.")
        st.stop()
    version = data_version(league)
    with st.status("Analysing the match...", expanded=True) as status:
        st.write("1 · Pulling facts with SQL: table, form, home/away record, head-to-head")
        st.write("2 · Retrieving similar past matches (RAG) and fitting the team-strength model")
        t0 = time.perf_counter()
        ctx = context(home, away, kickoff, league, season, version)
        t1 = time.perf_counter()
        st.write("3 · Writing the report and fact checking every number")
        result = report(home, away, kickoff, league, season, version, _ai_allowed())
        t2 = time.perf_counter()
        status.update(label="Analysis ready", state="complete", expanded=False)
    # wall-clock of this click (0 s when served from cache), shown so slow steps are visible
    st.session_state["analysis"] = (ctx, result, {"data": t1 - t0, "report": t2 - t1})

if "analysis" not in st.session_state:
    st.stop()

ctx, result, timing = st.session_state["analysis"]
home, away = ctx["match"]["home"], ctx["match"]["away"]
st.write("")
ui.html_block(ui.matchup(home, away, ctx["model"], ctx["match"]["date"]))
st.write("")

left, right = st.columns(2)
left.markdown(ui.team_card(home, ui.C["home"], ctx["table"][home], ctx["form"][home], ctx["home_record"], "Home"),
              unsafe_allow_html=True)
right.markdown(ui.team_card(away, ui.C["away"], ctx["table"][away], ctx["form"][away], ctx["away_record"], "Away"),
               unsafe_allow_html=True)
st.write("")

hh = ctx["head_to_head"]
if hh["meetings"]:
    total = hh["meetings"]
    seg = lambda n, color: f'<div style="width:{100 * n / total}%;background:{color}"></div>'
    recent = ui.games_html(hh["recent"])
    ui.html_block(f"""
<div class="card"><h4>Head-to-head · {total} meetings</h4>
  <div class="bar">{seg(hh[f'{home}_wins'], ui.C['home'])}{seg(hh['draws'], ui.C['draw'])}{seg(hh[f'{away}_wins'], ui.C['away'])}</div>
  <div class="legend"><div>{ui.e(home)}<b>{hh[f'{home}_wins']}</b></div><div style="text-align:center">Draws<b>{hh['draws']}</b></div>
  <div style="text-align:right">{ui.e(away)}<b>{hh[f'{away}_wins']}</b></div></div>
  <h4 style="margin-top:16px">Most recent</h4><div class="games">{recent}</div>
</div>""")
    st.write("")

v = result["validation"]
mode = result["mode"].split(":")  # e.g. llm:gemini:gemini-flash-lite-latest
writer = {"llm:gemini": "Gemini", "llm:claude": "Claude"}.get(":".join(mode[:2]), "the offline template")
if len(mode) == 3 and mode[0] == "llm":
    writer += f" ({mode[2]})"
ui.html_block(f'<div class="check {"" if v["passed"] else "bad"}">'
              f'{"✓ Every number in the report was checked against the data" if v["passed"] else "⚠ " + ui.e(", ".join(v["issues"]))}'
              f' · written by {ui.e(writer)}</div>'
              '<div class="hint">The fact checker is plain code, not AI: it compares every number and score in the '
              'report with the data above and sends the report back for a rewrite if one is invented.</div>'
              f'<div class="hint">Timing · data + model {timing["data"]:.1f} s · report {timing["report"]:.1f} s '
              f'({result["attempts"]} attempt{"s" if result["attempts"] > 1 else ""})</div>')
st.write("")

english, _, arabic = result["report"].partition("## العربية")  # no crash if a section is missing
tab_ar, tab_en, tab_rag = st.tabs(["التقرير بالعربي", "English report", "Retrieved matches (RAG)"])
tab_ar.markdown(ui.report_html(arabic, rtl=True), unsafe_allow_html=True)
tab_en.markdown(ui.report_html(english.replace("## English", "")), unsafe_allow_html=True)

def rag_row(i: int, doc: str) -> str:
    """'2026-10-30 A 1-1 B, red cards 0-1.' -> numbered row: date · result in bold · extra stats muted"""
    head, _, details = doc.rstrip(".").partition(", ")
    return (f'<div class="rag"><span class="i">#{i}</span><span class="d" style="color:{ui.C["muted"]};min-width:96px">'
            f'{ui.e(ui.nice_date(head[:10])[4:])}</span><span><b>{ui.e(head[11:])}</b>'
            f'<span style="color:{ui.C["muted"]}"> {ui.e(details)}</span></span></div>')


docs = ctx["retrieved_documents"]
tab_rag.markdown('<div class="card"><h4>Past matches the AI read before writing, most similar first</h4>'
                 + ("".join(rag_row(i, d) for i, d in enumerate(docs, 1)) or '<div class="rag">No similar matches found.</div>')
                 + "</div>", unsafe_allow_html=True)
