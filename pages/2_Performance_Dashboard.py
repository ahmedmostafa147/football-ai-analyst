"""Team Performance Dashboard for the current season: team analysis, league overview, strengths & weaknesses."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

import ui
from pipeline import LEAGUES, engine
from retrieval import seasons

ui.setup("Performance Dashboard", "📊")

STATS = ["goals", "xg", "shots", "sot", "corners", "yellow", "red"]
METRICS = {  # column: (label, higher is better)
    "goals_for": ("Goals scored", True), "goals_against": ("Goals conceded", False),
    "xg_for": ("Expected goals (xG)", True), "xg_against": ("xG conceded", False),
    "shots_for": ("Shots", True), "shots_against": ("Shots conceded", False),
    "sot_for": ("Shots on target", True), "sot_against": ("Shots on target conceded", False),
    "corners_for": ("Corners won", True), "yellow_for": ("Yellow cards", False),
    "points": ("Points per match", True), "clean_sheet": ("Clean sheets %", True),
}


@st.cache_data
def team_matches(league: str, season: str) -> pd.DataFrame:
    """One row per team per match, from that team's point of view."""
    with engine().connect() as c:
        df = pd.read_sql(text("SELECT * FROM matches WHERE league = :l AND season = :s"), c,
                         params={"l": league, "s": season})
    for side in ("home", "away"):
        for s in STATS:
            df[f"{side}_{s}"] = pd.to_numeric(df[f"{side}_{s}"], errors="coerce")
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        part = pd.DataFrame({"date": df.date, "team": df[side], "opponent": df[other], "venue": side.title()})
        for s in STATS:
            part[f"{s}_for"], part[f"{s}_against"] = df[f"{side}_{s}"], df[f"{other}_{s}"]
        rows.append(part)
    tm = pd.concat(rows, ignore_index=True)
    tm["points"] = (tm.goals_for > tm.goals_against) * 3 + (tm.goals_for == tm.goals_against) * 1
    tm["clean_sheet"] = (tm.goals_against == 0) * 100
    tm["result"] = tm.points.map({3: "W", 1: "D", 0: "L"})
    return tm.sort_values("date")


ui.hero("Team performance", "Performance Dashboard",
        "Every team in the league, this season: attack, defence, chance quality and where each team "
        "is stronger or weaker than the rest of the league.")

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    league = c1.segmented_control("League", list(LEAGUES), format_func=LEAGUES.get, default="EGY") or "EGY"
    season = c2.selectbox("Season", seasons(league)[::-1], format_func=lambda s: f"20{s[:2]}/{s[2:]}")

tm = team_matches(league, season)
available = [m for m in METRICS if tm[m].notna().any()]  # Egyptian data has goals only
per_team = tm.groupby("team")[available].mean()
avg = per_team.mean()
table = tm.groupby("team").agg(played=("points", "size"), points=("points", "sum"), gf=("goals_for", "sum"),
                               ga=("goals_against", "sum"))
table = table.assign(gd=table.gf - table.ga).sort_values(["points", "gd", "gf"], ascending=False)
table.insert(0, "pos", range(1, len(table) + 1))
teams = list(table.index)
default_team = next((t for t in ("El Zamalek", "Zamalek SC", "Arsenal") if t in teams), teams[0])
st.caption(f"{len(tm) // 2} matches played this season · last match {tm.date.max()}")

tab_team, tab_league, tab_sw = st.tabs(["Team analysis", "League overview", "Strengths & weaknesses"])

# ---------- Team analysis ----------
with tab_team:
    team = st.selectbox("Team", teams, index=teams.index(default_team))
    games = tm[tm.team == team]
    row = per_team.loc[team]
    has_xg = "xg_for" in available
    tiles = [
        ui.tile("League position", f"{table.loc[team, 'pos']} of {len(table)}"),
        ui.tile("Points per match", f"{row.points:.2f}", row.points - avg.points),
        ui.tile("Goals scored / match", f"{row.goals_for:.2f}", row.goals_for - avg.goals_for),
        ui.tile("Goals conceded / match", f"{row.goals_against:.2f}", row.goals_against - avg.goals_against, good_when_up=False),
    ]
    if has_xg:
        tiles.append(ui.tile("xG / match", f"{row.xg_for:.2f}", row.xg_for - avg.xg_for))
    else:
        tiles.append(ui.tile("Clean sheets", f"{row.clean_sheet:.0f}%", row.clean_sheet - avg.clean_sheet))
    ui.tiles(tiles)
    st.write("")

    label = games.opponent.str.slice(0, 14) + " (" + games.venue.str[0] + ")"
    left, right = st.columns(2)
    fig = go.Figure([go.Bar(name="Scored", x=label, y=games.goals_for, marker_color=ui.C["accent"]),
                     go.Bar(name="Conceded", x=label, y=games.goals_against, marker_color=ui.C["loss"])])
    if has_xg:
        fig.add_trace(go.Scatter(name="xG created", x=label, y=games.xg_for, mode="lines+markers",
                                 line={"color": ui.C["home"], "width": 2}))
    fig.update_layout(barmode="group", title="Every match this season", height=380)
    left.plotly_chart(fig, use_container_width=True)

    cum = games.points.cumsum().reset_index(drop=True)
    fig = go.Figure(go.Scatter(x=list(range(1, len(cum) + 1)), y=cum, mode="lines+markers",
                               line={"color": ui.C["accent"], "width": 3}, fill="tozeroy",
                               fillcolor="rgba(43,217,160,0.12)"))
    fig.update_layout(title="Points collected over the season", height=380,
                      xaxis_title="Match number", yaxis_title="Points")
    right.plotly_chart(fig, use_container_width=True)

    cols = ["date", "venue", "opponent", "result", "goals_for", "goals_against"] + \
           [c for c in ("xg_for", "xg_against", "shots_for", "sot_for") if c in available]
    st.dataframe(games[cols].iloc[::-1], hide_index=True, use_container_width=True,
                 column_config={"goals_for": "Scored", "goals_against": "Conceded", "xg_for": "xG",
                                "xg_against": "xG against", "shots_for": "Shots", "sot_for": "On target"})

# ---------- League overview ----------
with tab_league:
    x, y = ("xg_for", "xg_against") if "xg_for" in available else ("goals_for", "goals_against")
    fig = go.Figure(go.Scatter(x=per_team[x], y=per_team[y], mode="markers+text", text=per_team.index,
                               textposition="top center", textfont={"size": 11},
                               marker={"size": 13, "color": [ui.C["accent"] if t == default_team else ui.C["home"]
                                                              for t in per_team.index]}))
    fig.add_vline(x=avg[x], line_dash="dot", line_color=ui.C["muted"])
    fig.add_hline(y=avg[y], line_dash="dot", line_color=ui.C["muted"])
    fig.update_layout(title=f"Attack vs defence per match (bottom right = best)", height=560,
                      xaxis_title=METRICS[x][0] + " per match", yaxis_title=METRICS[y][0] + " per match",
                      yaxis={"autorange": "reversed"})
    left, right = st.columns([3, 2])
    left.plotly_chart(fig, use_container_width=True)
    right.dataframe(table, use_container_width=True, height=560,
                    column_config={"pos": "#", "played": "P", "points": "Pts", "gf": "GF", "ga": "GA", "gd": "GD"})

# ---------- Strengths & weaknesses ----------
with tab_sw:
    team2 = st.selectbox("Team ", teams, index=teams.index(default_team))
    z = (per_team.loc[team2] - avg) / per_team.std()
    z = pd.Series({m: (v if METRICS[m][1] else -v) for m, v in z.items()}).dropna()  # positive = better
    ranked = z.sort_values()
    unit = lambda m: "%" if m == "clean_sheet" else " per match"

    def items(metrics, empty: str) -> str:
        if not len(metrics):
            return f'<div class="list-item"><span class="x">{empty}</span></div>'
        return "".join(f'<div class="list-item"><span class="n">{METRICS[m][0]}</span>'
                       f'<span class="x">{per_team.loc[team2, m]:.2f}{unit(m)} · league {avg[m]:.2f}</span></div>'
                       for m in metrics)

    strengths = ranked[ranked > 0].index[::-1][:3]  # only metrics where the team is better than average
    weaknesses = ranked[ranked < 0].index[:3]
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="card" style="border-top:3px solid {ui.C["win"]}"><h4>Strengths</h4>'
                f'{items(strengths, "No metric above the league average")}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card" style="border-top:3px solid {ui.C["loss"]}"><h4>Weaknesses</h4>'
                f'{items(weaknesses, "No metric below the league average")}</div>', unsafe_allow_html=True)
    st.write("")
    fig = go.Figure(go.Bar(x=ranked.values, y=[METRICS[m][0] for m in ranked.index], orientation="h",
                           marker_color=[ui.C["win"] if v > 0 else ui.C["loss"] for v in ranked.values]))
    fig.update_layout(title=f"{team2} vs league average (standard deviations, right = better)", height=440)
    st.plotly_chart(fig, use_container_width=True)
