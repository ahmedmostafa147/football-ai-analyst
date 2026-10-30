"""Offline report writer (no API key needed). Same structure the AI writer is asked to follow."""


def _pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _iso(s) -> str:
    return f"⁨{s}⁩"  # keeps English names and scores intact inside Arabic text


def template_report(ctx: dict) -> str:
    h, a = ctx["match"]["home"], ctx["match"]["away"]
    t, f, m, hh = ctx["table"], ctx["form"], ctx["model"], ctx["head_to_head"]
    hr, ar = ctx["home_record"], ctx["away_record"]
    pos = lambda team: t[team]["position"] if t[team] else "-"
    pts = lambda team: t[team]["pts"] if t[team] else "-"
    probs = {"home": m["p_home_win"], "draw": m["p_draw"], "away": m["p_away_win"]}
    fav = max(probs, key=probs.get)
    gap = sorted(probs.values())[-1] - sorted(probs.values())[-2]
    tight = gap < 0.08

    # ---------------- English ----------------
    headline_en = ("A tight game: the model gives the top two outcomes almost the same chance."
                   if tight else
                   {"home": f"{h} are favourites at home ({_pct(m['p_home_win'])}).",
                    "draw": f"The model's most likely outcome is a draw ({_pct(m['p_draw'])}).",
                    "away": f"{a} are favourites away from home ({_pct(m['p_away_win'])})."}[fav])

    def record_en(team, r, venue):
        if not r.get("played"):
            return f"- {team} have no {venue} matches yet this season."
        line = f"- {team} {venue}: {r['scored']} scored and {r['conceded']} conceded per game"
        if r.get("xg_for") is not None:
            line += f", xG {r['xg_for']} for and {r['xg_against']} against"
        if r.get("shots_on_target") is not None:
            line += f", {r['shots_on_target']} shots on target"
        return line + "."

    lines_en = [
        "## English",
        "### Headline", f"- {headline_en}",
        "### Current form",
        f"- {h}: position {pos(h)}, {pts(h)} points, last five {f[h]['last_results']} ({f[h]['points']} points).",
        f"- {a}: position {pos(a)}, {pts(a)} points, last five {f[a]['last_results']} ({f[a]['points']} points).",
        "### Key matchup", record_en(h, hr, "at home"), record_en(a, ar, "away"),
    ]
    if hh["meetings"]:
        lines_en += ["### History",
                     f"- {hh['meetings']} meetings: {h} won {hh[f'{h}_wins']}, {hh['draws']} draws, {a} won {hh[f'{a}_wins']}.",
                     f"- Latest: {hh['recent'][0]}."]
    lines_en += [
        "### Prediction",
        f"- {h} win {_pct(m['p_home_win'])} · draw {_pct(m['p_draw'])} · {a} win {_pct(m['p_away_win'])}.",
        f"- Expected goals {m['expected_goals_home']} - {m['expected_goals_away']}, most likely score {m['most_likely_score']}.",
        "### Verdict",
        f"- {'Too close to call, with a slight edge to ' if tight else 'The numbers point to '}"
        f"{ {'home': h, 'draw': 'a draw', 'away': a}[fav] }.",
    ]

    # ---------------- Arabic ----------------
    H, A = _iso(h), _iso(a)
    form_ar = lambda team: " · ".join({"W": "فوز", "D": "تعادل", "L": "خسارة"}[x] for x in f[team]["last_results"])
    headline_ar = ("مباراة متكافئة: النموذج يعطي أقرب نتيجتين فرصًا شبه متساوية."
                   if tight else
                   {"home": f"{H} مرشح للفوز على أرضه بنسبة {_iso(_pct(m['p_home_win']))}.",
                    "draw": f"التعادل هو النتيجة الأرجح بنسبة {_iso(_pct(m['p_draw']))}.",
                    "away": f"{A} مرشح للفوز خارج أرضه بنسبة {_iso(_pct(m['p_away_win']))}."}[fav])

    def record_ar(team, r, venue):
        if not r.get("played"):
            return f"- لم يلعب {team} أي مباراة {venue} هذا الموسم."
        line = f"- {team} {venue}: يسجل {r['scored']} ويستقبل {r['conceded']} في المباراة"
        if r.get("xg_for") is not None:
            line += f"، أهداف متوقعة {r['xg_for']} له و{r['xg_against']} عليه"
        if r.get("shots_on_target") is not None:
            line += f"، {r['shots_on_target']} تسديدة على المرمى"
        return line + "."

    lines_ar = [
        "## العربية",
        "### الخلاصة السريعة", f"- {headline_ar}",
        "### الفورمة الحالية",
        f"- {H}: المركز {pos(h)} برصيد {pts(h)} نقطة. آخر 5 مباريات: {form_ar(h)}.",
        f"- {A}: المركز {pos(a)} برصيد {pts(a)} نقطة. آخر 5 مباريات: {form_ar(a)}.",
        "### مفتاح المباراة", record_ar(H, hr, "على أرضه"), record_ar(A, ar, "خارج أرضه"),
    ]
    if hh["meetings"]:
        lines_ar += ["### التاريخ",
                     f"- {hh['meetings']} مواجهة: فاز {H} {hh[f'{h}_wins']}، تعادل {hh['draws']}، فاز {A} {hh[f'{a}_wins']}.",
                     f"- آخر مواجهة: {_iso(hh['recent'][0])}."]
    winner_ar = {"home": f"فوز {H}", "draw": "التعادل", "away": f"فوز {A}"}[fav]
    lines_ar += [
        "### التوقع",
        f"- فوز {H} {_iso(_pct(m['p_home_win']))} · تعادل {_iso(_pct(m['p_draw']))} · فوز {A} {_iso(_pct(m['p_away_win']))}.",
        f"- الأهداف المتوقعة {_iso(str(m['expected_goals_home']) + ' - ' + str(m['expected_goals_away']))}، "
        f"والنتيجة الأرجح {_iso(m['most_likely_score'])}.",
        "### الحكم النهائي",
        "- " + (("مباراة صعبة التوقع مع أفضلية بسيطة " + {"home": f"لصالح {H}", "draw": "للتعادل", "away": f"لصالح {A}"}[fav])
                if tight else f"الأرقام تميل إلى {winner_ar}") + ".",
    ]
    return "\n".join(lines_en) + "\n\n" + "\n".join(lines_ar) + "\n"
