"""Football Performance Dashboard: team analysis, player leaders, player comparison, strengths & weaknesses."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

from pipeline import engine

st.set_page_config(page_title="Performance Dashboard", page_icon="📊", layout="wide")
st.title("📊 Football Performance Dashboard")
st.caption("Team and player performance from StatsBomb open event data · every pass, shot and tackle")

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

competition = st.radio("Tournament", sorted(teams.competition.unique()), horizontal=True)
t_comp, p_comp = teams[teams.competition == competition], players[players.competition == competition]
p_qual = p_comp[p_comp.minutes >= MIN_MINUTES]
team_avg = t_comp.groupby("team")[list(TEAM_METRICS)].mean()
league_avg = team_avg.mean()

tab_team, tab_leaders, tab_compare, tab_sw = st.tabs(
    ["Team analysis", "Player leaders", "Player comparison", "Strengths & weaknesses"])

# ---------- Team analysis ----------
with tab_team:
    team_names = sorted(t_comp.team.unique())
    default = next((t for t in ("Egypt", "Morocco", "Saudi Arabia") if t in team_names), team_names[0])
    team = st.selectbox("Team", team_names, index=team_names.index(default))
    games = t_comp[t_comp.team == team].merge(matches[["match_id", "date", "stage"]], on="match_id").sort_values("date")
    k = st.columns(5)
    for col, (metric, fmt) in zip(k, [("goals", "{:.2f}"), ("xg", "{:.2f}"), ("shots", "{:.1f}"),
                                      ("possession", "{:.0f}%"), ("pass_pct", "{:.0f}%")]):
        value, avg = team_avg.loc[team, metric], league_avg[metric]
        col.metric(f"{TEAM_METRICS[metric][0]} / match", fmt.format(value), f"{value - avg:+.2f} vs avg")

    compare = pd.DataFrame({"metric": [TEAM_METRICS[m][0] for m in ["goals", "xg", "shots", "shots_on_target", "key_passes"]] * 2,
                            "value": list(team_avg.loc[team, ["goals", "xg", "shots", "shots_on_target", "key_passes"]])
                            + list(league_avg[["goals", "xg", "shots", "shots_on_target", "key_passes"]]),
                            "who": [team] * 5 + ["Tournament average"] * 5})
    st.plotly_chart(px.bar(compare, x="metric", y="value", color="who", barmode="group",
                           title=f"{team} attack per match vs tournament average"), use_container_width=True)

    games["match"] = games.date + " vs " + games.opponent
    fig = go.Figure([go.Bar(name="xG for", x=games.match, y=games.xg), go.Bar(name="xG against", x=games.match, y=games.xg_against)])
    fig.update_layout(barmode="group", title="Chance quality in each match (xG)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(games[["date", "stage", "opponent", "goals", "goals_against", "xg", "xg_against", "shots",
                        "possession", "pass_pct"]].round(2), hide_index=True, use_container_width=True)

# ---------- Player leaders ----------
with tab_leaders:
    st.caption(f"Per 90 minutes · players with at least {MIN_MINUTES} minutes")
    c1, c2 = st.columns(2)
    for col, metric, title in ((c1, "key_passes_90", "Best chance creators"), (c2, "shots_90", "Most shots")):
        top = p_qual.nlargest(10, metric).sort_values(metric)
        col.plotly_chart(px.bar(top, x=metric, y="player", color="team", orientation="h", title=title,
                                labels={metric: PLAYER_METRICS[metric], "player": ""}), use_container_width=True)
    c3, c4 = st.columns(2)
    for col, metric, title in ((c3, "xg_90", "Highest expected goals"), (c4, "tackles_won_90", "Best ball winners")):
        top = p_qual.nlargest(10, metric).sort_values(metric)
        col.plotly_chart(px.bar(top, x=metric, y="player", color="team", orientation="h", title=title,
                                labels={metric: PLAYER_METRICS[metric], "player": ""}), use_container_width=True)

# ---------- Player comparison ----------
with tab_compare:
    group = st.selectbox("Position", ["Forward", "Midfielder", "Defender"])
    pool = p_qual[p_qual.group == group]
    names = sorted(pool.player)
    if len(names) < 2:
        st.info("Not enough players in this position.")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Player A", names, index=0)
        b = c2.selectbox("Player B", names, index=1)
        metrics = list(PLAYER_METRICS)
        pct = pool.set_index("player")[metrics].rank(pct=True) * 100  # percentile within the same position
        fig = go.Figure()
        for name in (a, b):
            fig.add_trace(go.Scatterpolar(r=list(pct.loc[name]) + [pct.loc[name].iloc[0]],
                                          theta=[PLAYER_METRICS[m] for m in metrics] + [PLAYER_METRICS[metrics[0]]],
                                          fill="toself", name=name))
        fig.update_layout(polar={"radialaxis": {"range": [0, 100]}},
                          title=f"Percentile vs all {group.lower()}s in {competition}")
        st.plotly_chart(fig, use_container_width=True)
        table = pool.set_index("player").loc[[a, b], ["team", "minutes", *metrics]]
        st.dataframe(table.rename(columns=PLAYER_METRICS).round(2).T, use_container_width=True)

# ---------- Strengths & weaknesses ----------
with tab_sw:
    team2 = st.selectbox("Team ", sorted(t_comp.team.unique()), index=sorted(t_comp.team.unique()).index(default))
    z = (team_avg.loc[team2] - league_avg) / team_avg.std()
    z = pd.Series({m: (v if TEAM_METRICS[m][1] else -v) for m, v in z.items()})  # positive = good
    ranked = z.sort_values()
    c1, c2 = st.columns(2)
    c1.subheader("✅ Strengths")
    for m in ranked.index[::-1][:3]:
        c1.write(f"**{TEAM_METRICS[m][0]}**: {team_avg.loc[team2, m]:.2f} per match (avg {league_avg[m]:.2f})")
    c2.subheader("⚠️ Weaknesses")
    for m in ranked.index[:3]:
        c2.write(f"**{TEAM_METRICS[m][0]}**: {team_avg.loc[team2, m]:.2f} per match (avg {league_avg[m]:.2f})")
    chart = pd.DataFrame({"metric": [TEAM_METRICS[m][0] for m in ranked.index], "score": ranked.values})
    chart["side"] = chart.score.map(lambda s: "Strength" if s > 0 else "Weakness")
    st.plotly_chart(px.bar(chart, x="score", y="metric", color="side", orientation="h",
                           color_discrete_map={"Strength": "#1D9E75", "Weakness": "#D85A30"},
                           title=f"{team2} vs tournament average (standard deviations, positive = better)"),
                    use_container_width=True)
