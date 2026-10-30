"""Football Performance Dashboard: team analysis, player leaders, player comparison, strengths & weaknesses."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

import ui
from pipeline import engine

ui.setup("Performance Dashboard", "📊")

MIN_MINUTES = 180
TEAM_METRICS = {  # column: (label, higher is better)
    "goals": ("Goals", True), "xg": ("Expected goals (xG)", True), "shots": ("Shots", True),
    "shots_on_target": ("Shots on target", True), "possession": ("Possession %", True),
    "passes": ("Passes", True), "pass_pct": ("Pass accuracy %", True), "key_passes": ("Chances created", True),
    "dribbles_completed": ("Successful dribbles", True), "tackles_won": ("Tackles won", True),
    "interceptions": ("Interceptions", True), "goals_against": ("Goals conceded", False),
    "xg_against": ("xG conceded", False), "shots_against": ("Shots conceded", False),
}
PLAYER_METRICS = {  # per-90 column: label
    "goals_90": "Goals", "xg_90": "xG", "shots_90": "Shots", "key_passes_90": "Chances created",
    "assists_90": "Assists", "dribbles_completed_90": "Dribbles", "pass_pct": "Pass accuracy %",
    "tackles_won_90": "Tackles won", "interceptions_90": "Interceptions", "recoveries_90": "Ball recoveries",
}


def position_group(pos: str) -> str:
    if pos == "Goalkeeper":
        return "Goalkeeper"
    if "Back" in pos:
        return "Defender"
    if "Midfield" in pos:
        return "Midfielder"
    if any(w in pos for w in ("Forward", "Wing", "Striker")):
        return "Forward"
    return "Other"


@st.cache_data
def load():
    with engine().connect() as c:
        try:
            teams = pd.read_sql(text("SELECT * FROM sb_team_match"), c)
            players = pd.read_sql(text("SELECT * FROM sb_player_match"), c)
            matches = pd.read_sql(text("SELECT * FROM sb_matches"), c)
        except Exception:
            return None
    teams["pass_pct"] = 100 * teams.passes_completed / teams.passes.clip(lower=1)
    totals = players.groupby(["competition", "player", "team"], as_index=False).agg(
        position=("position", lambda s: s.mode().iat[0]), matches=("match_id", "nunique"),
        **{c: (c, "sum") for c in ["minutes", "goals", "xg", "shots", "shots_on_target", "key_passes", "assists",
                                   "passes", "passes_completed", "dribbles_completed", "tackles_won",
                                   "interceptions", "recoveries"]})
    for col in ["goals", "xg", "shots", "key_passes", "assists", "dribbles_completed", "tackles_won",
                "interceptions", "recoveries"]:
        totals[f"{col}_90"] = 90 * totals[col] / totals.minutes.clip(lower=1)
    totals["pass_pct"] = 100 * totals.passes_completed / totals.passes.clip(lower=1)
    totals["group"] = totals.position.map(position_group)
    return teams, totals, matches


data = load()
if data is None:
    st.error("No performance data yet. Run `python statsbomb.py` first.")
    st.stop()
teams, players, matches = data

ui.hero("Performance analysis", "Performance Dashboard",
        "Team and player performance from StatsBomb event data: every pass, shot, dribble and tackle.",
        [f"{len(matches)} matches", f"{players.player.nunique():,} players", "AFCON 2023 · World Cup 2022 · Euro 2024"])

competition = st.segmented_control("Tournament", sorted(teams.competition.unique()), default="AFCON 2023") or "AFCON 2023"
t_comp, p_comp = teams[teams.competition == competition], players[players.competition == competition]
p_qual = p_comp[p_comp.minutes >= MIN_MINUTES]
team_avg = t_comp.groupby("team")[list(TEAM_METRICS)].mean()
league_avg = team_avg.mean()
team_names = sorted(t_comp.team.unique())
default_team = next((t for t in ("Egypt", "Morocco", "Saudi Arabia") if t in team_names), team_names[0])

tab_team, tab_leaders, tab_compare, tab_sw = st.tabs(
    ["Team analysis", "Player leaders", "Player comparison", "Strengths & weaknesses"])

# ---------- Team analysis ----------
with tab_team:
    team = st.selectbox("Team", team_names, index=team_names.index(default_team))
    games = t_comp[t_comp.team == team].merge(matches[["match_id", "date", "stage"]], on="match_id").sort_values("date")
    row = team_avg.loc[team]
    ui.tiles([
        ui.tile("Goals per match", f"{row.goals:.2f}", row.goals - league_avg.goals),
        ui.tile("xG per match", f"{row.xg:.2f}", row.xg - league_avg.xg),
        ui.tile("Shots per match", f"{row.shots:.1f}", row.shots - league_avg.shots),
        ui.tile("Possession", f"{row.possession:.0f}%", row.possession - league_avg.possession),
        ui.tile("Goals conceded", f"{row.goals_against:.2f}", row.goals_against - league_avg.goals_against, good_when_up=False),
    ])
    st.write("")
    c1, c2 = st.columns(2)
    keys = ["goals", "xg", "shots_on_target", "key_passes"]
    fig = go.Figure([
        go.Bar(name=team, x=[TEAM_METRICS[k][0] for k in keys], y=row[keys], marker_color=ui.C["accent"]),
        go.Bar(name="Tournament average", x=[TEAM_METRICS[k][0] for k in keys], y=league_avg[keys], marker_color=ui.C["border"]),
    ])
    fig.update_layout(barmode="group", title="Attack per match vs tournament average", height=360)
    c1.plotly_chart(fig, use_container_width=True)

    label = games.opponent.str.slice(0, 14)
    fig = go.Figure([
        go.Bar(name="xG created", x=label, y=games.xg, marker_color=ui.C["accent"]),
        go.Bar(name="xG conceded", x=label, y=games.xg_against, marker_color=ui.C["loss"]),
    ])
    fig.update_layout(barmode="group", title="Chance quality in each match", height=360)
    c2.plotly_chart(fig, use_container_width=True)

    table = games.assign(result=games.goals.astype(int).astype(str) + "–" + games.goals_against.astype(int).astype(str))
    st.dataframe(
        table[["date", "stage", "opponent", "result", "xg", "xg_against", "shots", "possession", "pass_pct"]].round(1),
        hide_index=True, use_container_width=True,
        column_config={"xg": "xG", "xg_against": "xG against", "pass_pct": "Pass %", "possession": "Possession %"})

# ---------- Player leaders ----------
with tab_leaders:
    st.caption(f"Per 90 minutes · players with at least {MIN_MINUTES} minutes")

    def short(name: str) -> str:  # "Mohamed Hany Gamal Eldemerdash" -> "Mohamed Eldemerdash"
        words = name.split()
        return name if len(name) <= 20 or len(words) < 3 else f"{words[0]} {words[-1]}"

    def leader_chart(metric: str, title: str, color: str):
        top = p_qual.nlargest(10, metric).sort_values(metric)
        labels = [f"{short(p)} <span style='color:{ui.C['muted']}'>· {t}</span>" for p, t in zip(top.player, top.team)]
        fig = go.Figure(go.Bar(x=top[metric], y=labels, orientation="h", marker_color=color,
                               text=top[metric].round(2), textposition="outside", cliponaxis=False,
                               hovertext=top.player, hoverinfo="text+x"))
        fig.update_layout(title=title, height=420, xaxis={"range": [0, top[metric].max() * 1.18], "showgrid": False,
                                                          "showticklabels": False},
                          margin={"l": 10, "r": 30, "t": 48, "b": 10})
        return fig

    c1, c2 = st.columns(2)
    c1.plotly_chart(leader_chart("key_passes_90", "Best chance creators", ui.C["accent"]), use_container_width=True)
    c2.plotly_chart(leader_chart("shots_90", "Most shots", ui.C["home"]), use_container_width=True)
    c3, c4 = st.columns(2)
    c3.plotly_chart(leader_chart("xg_90", "Highest expected goals", ui.C["away"]), use_container_width=True)
    c4.plotly_chart(leader_chart("tackles_won_90", "Best ball winners", "#B07CFF"), use_container_width=True)

# ---------- Player comparison ----------
with tab_compare:
    group = st.segmented_control("Position", ["Forward", "Midfielder", "Defender"], default="Forward") or "Forward"
    pool = p_qual[p_qual.group == group].sort_values("xg_90", ascending=False)
    names = list(pool.player)
    if len(names) < 2:
        st.info("Not enough players in this position.")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Player A", names, index=0)
        b = c2.selectbox("Player B", names, index=1)
        metrics = list(PLAYER_METRICS)
        pct = pool.set_index("player")[metrics].rank(pct=True) * 100  # percentile within the same position
        fig = go.Figure()
        for name, color in ((a, ui.C["home"]), (b, ui.C["away"])):
            r = list(pct.loc[name]) + [pct.loc[name].iloc[0]]
            fig.add_trace(go.Scatterpolar(r=r, theta=[PLAYER_METRICS[m] for m in metrics] + [PLAYER_METRICS[metrics[0]]],
                                          fill="toself", name=name, line_color=color, opacity=0.75))
        fig.update_layout(polar={"radialaxis": {"range": [0, 100], "showticklabels": False}}, height=480,
                          title=f"Percentile vs all {group.lower()}s in {competition}")
        left, right = st.columns([3, 2])
        left.plotly_chart(fig, use_container_width=True)
        table = pool.set_index("player").loc[[a, b], ["team", "minutes", *metrics]]
        table.index = [short(a), short(b)]
        right.dataframe(table.rename(columns=PLAYER_METRICS).round(2).T, use_container_width=True, height=470)

# ---------- Strengths & weaknesses ----------
with tab_sw:
    team2 = st.selectbox("Team ", team_names, index=team_names.index(default_team))
    z = (team_avg.loc[team2] - league_avg) / team_avg.std()
    z = pd.Series({m: (v if TEAM_METRICS[m][1] else -v) for m, v in z.items()})  # positive = good
    ranked = z.sort_values()

    def items(metrics, empty: str) -> str:
        if not len(metrics):
            return f'<div class="list-item"><span class="x">{empty}</span></div>'
        return "".join(
            f'<div class="list-item"><span class="n">{TEAM_METRICS[m][0]}</span>'
            f'<span class="x">{team_avg.loc[team2, m]:.2f} per match · avg {league_avg[m]:.2f}</span></div>'
            for m in metrics)

    strengths = ranked[ranked > 0].index[::-1][:3]  # only metrics where the team is better than average
    weaknesses = ranked[ranked < 0].index[:3]
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="card" style="border-top:3px solid {ui.C["win"]}"><h4>Strengths</h4>'
                f'{items(strengths, "No metric above the tournament average")}</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card" style="border-top:3px solid {ui.C["loss"]}"><h4>Weaknesses</h4>'
                f'{items(weaknesses, "No metric below the tournament average")}</div>', unsafe_allow_html=True)
    st.write("")
    fig = go.Figure(go.Bar(x=ranked.values, y=[TEAM_METRICS[m][0] for m in ranked.index], orientation="h",
                           marker_color=[ui.C["win"] if v > 0 else ui.C["loss"] for v in ranked.values]))
    fig.update_layout(title=f"{team2} vs tournament average (standard deviations, right = better)", height=480)
    st.plotly_chart(fig, use_container_width=True)
