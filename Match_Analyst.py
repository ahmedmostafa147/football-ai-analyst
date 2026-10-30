"""Web page: streamlit run Match_Analyst.py"""
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import text

from agents import orchestrate
from pipeline import LEAGUES, engine
from retrieval import build_context, seasons, teams

# default matchup per league (names differ between the Kaggle and footballdatabase sources)
DEFAULTS = {"EPL": (["Arsenal"], ["Liverpool"]),
            "EGY": (["El Zamalek", "Zamalek SC"], ["Al Ahly", "El Ahly Cairo"])}


def _pick(names: list[str], options: list[str], fallback: int) -> int:
    return next((options.index(n) for n in names if n in options), fallback)

st.set_page_config(page_title="Football AI Match Analyst", page_icon="⚽", layout="wide")
st.title("⚽ Football AI Match Analyst")
st.caption("Pre-match analysis from real match data · SQL + RAG + LLM + fact checking")

l1, l2 = st.columns([3, 1])
league = l1.radio("League", list(LEAGUES), format_func=LEAGUES.get, horizontal=True)
season_list = seasons(league)
season = l2.selectbox("Season", season_list[::-1], format_func=lambda s: f"20{s[:2]}/{s[2:]}")
with engine().connect() as c:
    n_matches = c.execute(text("SELECT COUNT(*) FROM matches WHERE league = :l"), {"l": league}).scalar()
st.caption(f"{n_matches:,} matches in the database · seasons 20{season_list[0][:2]} → 20{season_list[-1][2:]}")

team_list = teams(league, season)
with engine().connect() as c:
    last_played = c.execute(text("SELECT MAX(date) FROM matches WHERE league = :l AND season = :s"),
                            {"l": league, "s": season}).scalar()
d_home, d_away = DEFAULTS[league]
c1, c2, c3 = st.columns([2, 2, 1])
home = c1.selectbox("Home team", team_list, index=_pick(d_home, team_list, 0))
away = c2.selectbox("Away team", team_list, index=_pick(d_away, team_list, 1))
kickoff = c3.date_input("Match date", date.fromisoformat(str(last_played)) + timedelta(days=1))

if st.button("Analyse match", type="primary", use_container_width=True):
    if home == away:
        st.error("Pick two different teams.")
        st.stop()
    with st.spinner("Collecting data, analysing, fact checking..."):
        ctx = build_context(home, away, kickoff, league, season)
        result = orchestrate(ctx)

    m = ctx["model"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"{home} win", f"{m['p_home_win'] * 100:.1f}%")
    k2.metric("Draw", f"{m['p_draw'] * 100:.1f}%")
    k3.metric(f"{away} win", f"{m['p_away_win'] * 100:.1f}%")
    k4.metric("Most likely score", m["most_likely_score"])

    left, right = st.columns(2)
    for col, team in ((left, home), (right, away)):
        f, t = ctx["form"][team], ctx["table"][team]
        col.subheader(team)
        col.write(f"**Position:** {t['position'] if t else '-'} · **Points:** {t['pts'] if t else '-'} · **Form:** {f['last_results']}")
        col.dataframe(pd.DataFrame({"Last 5 matches": f["games"]}), hide_index=True, use_container_width=True)

    hh = ctx["head_to_head"]
    if hh["meetings"]:
        st.write(f"**Head-to-head ({hh['meetings']} meetings):** {home} {hh[f'{home}_wins']} wins · "
                 f"{hh['draws']} draws · {away} {hh[f'{away}_wins']} wins")

    v = result["validation"]
    if v["passed"]:
        st.success(f"Fact check passed · writer: {result['mode']}")
    else:
        st.warning(f"Fact check issues: {', '.join(v['issues'])}")

    english, arabic = result["report"].split("## العربية")
    tab_en, tab_ar = st.tabs(["English report", "التقرير بالعربي"])
    tab_en.markdown(english.replace("## English", ""))
    paragraphs = "".join(
        f"<p>{re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', p.strip())}</p>" for p in arabic.split("\n\n") if p.strip()
    )
    tab_ar.markdown(f"<div dir='rtl' style='text-align:right'>{paragraphs}</div>", unsafe_allow_html=True)

    with st.expander("Retrieved documents (RAG)"):
        for d in ctx["retrieved_documents"]:
            st.write("• " + d)
