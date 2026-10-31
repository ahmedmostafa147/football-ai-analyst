"""Analysis agent (LLM) + validation agent (fact checker), orchestrated with a retry loop."""
import json
import os
import re
import time

CLAUDE_MODEL = os.getenv("ANALYST_MODEL", "claude-sonnet-5-5")
# lite first: measured on the live site it wrote a report that passed the fact check in ~1.5 s,
# while gemini-flash-latest often took 25 s+ (once 158 s)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_FALLBACK = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-latest")
GEMINI_TIMEOUT_MS = 25_000

SYSTEM = """You are a senior football performance analyst writing a pre-match briefing for a coaching staff.
Use ONLY the facts in the JSON context. Never invent players, injuries, line-ups or numbers.
Every number you write must appear in the context (probabilities may be written as percentages,
e.g. 0.456 -> 45.6%). Do not repeat dates.

Write two sections with these exact headings, the Arabic one in clear Modern Standard Arabic:
## English
## العربية

Inside each section use exactly these sub-headings (### in markdown), each with 2-3 short bullet points:
### Headline        (one bullet: the single most important insight about this match)
### Current form    (table position, points, recent results)
### Key matchup     (home team at home vs away team away: goals scored/conceded, shots, xG when available)
### History         (head-to-head record, only if meetings > 0)
### Prediction      (model probabilities, expected goals, most likely score)
### Verdict         (one bullet, one sentence)
Arabic sub-headings: الخلاصة السريعة، الفورمة الحالية، مفتاح المباراة، التاريخ، التوقع، الحكم النهائي.
Interpret the numbers (what they mean on the pitch), don't just list them. Max 200 words per section."""


def _provider():
    """Gemini if GEMINI_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else the offline template."""
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    return None


class AnalysisAgent:
    def __init__(self):
        self.provider = _provider()
        self.error = None
        self.model_used = GEMINI_MODEL if self.provider == "gemini" else CLAUDE_MODEL

    @property
    def mode(self) -> str:
        if self.error:
            return f"template (AI writer failed: {self.error})"
        return {"gemini": f"llm:gemini:{self.model_used}", "claude": f"llm:claude:{CLAUDE_MODEL}"}.get(
            self.provider, "template (set GEMINI_API_KEY or ANTHROPIC_API_KEY for AI mode)")

    def _ask(self, prompt: str) -> str:
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            from google.genai import errors
            import httpx
            # 25 s per request: an overloaded model once kept a visitor waiting 158 s
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                                  http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
            # thinking tokens count towards max_output_tokens (3000 cut the Arabic section off mid-report);
            # a small thinking budget keeps the report fast, the numbers are already computed for it
            config = types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3, max_output_tokens=8192,
                                                 thinking_config=types.ThinkingConfig(thinking_budget=512))
            # main (fast) model twice, then the bigger model once if it is overloaded (503) or slow
            for model, tries in ((GEMINI_MODEL, 2), (GEMINI_FALLBACK, 1)):
                for attempt in range(tries):
                    try:
                        resp = client.models.generate_content(model=model, contents=prompt, config=config)
                        self.model_used = model
                        return resp.text or ""
                    except (errors.ServerError, httpx.TimeoutException):
                        if attempt + 1 < tries:
                            time.sleep(2)
            raise RuntimeError("Gemini overloaded")
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model=CLAUDE_MODEL, max_tokens=3000, system=SYSTEM, messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text

    def run(self, ctx: dict, feedback: list[str] | None = None) -> str:
        if not self.provider or self.error:
            return template_report(ctx)
        prompt = "Context:\n" + json.dumps(ctx, ensure_ascii=False, default=str)
        if feedback:
            prompt += "\n\nYour previous draft failed fact-checking. Fix these issues:\n- " + "\n- ".join(feedback)
        try:
            return self._ask(prompt)
        except Exception as e:  # bad key, quota, network: keep the site working with the template writer
            self.error = e.__class__.__name__
            return template_report(ctx)


class ValidationAgent:
    """Deterministic fact checker: every number and score in the report must be grounded in the context."""

    NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
    SCORE = re.compile(r"(?<![\d.])(\d+)\s*[-–]\s*(\d+)(?![\d.])")
    ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    REQUIRED_EN = ("Headline", "Current form", "Key matchup", "Prediction", "Verdict")  # History is optional
    REQUIRED_AR = ("الخلاصة السريعة", "الفورمة الحالية", "مفتاح المباراة", "التوقع", "الحكم النهائي")

    def allowed_numbers(self, ctx: dict) -> set[str]:
        raw = json.dumps(ctx, default=str)
        allowed = set(re.findall(r"\d+(?:\.\d+)?", raw))
        for v in list(allowed):
            f = float(v)
            allowed |= {f"{f:g}", str(round(f)), f"{f:.1f}", f"{f:.2f}"}
            if f <= 1:  # probability -> percentage forms
                allowed |= {f"{f * 100:.1f}%", f"{f * 100:.0f}%", f"{f * 100:g}%"}
        allowed |= {str(i) for i in range(0, 11)}  # small counts ("last 5", "2 wins") are ok
        return allowed

    def run(self, report: str, ctx: dict) -> dict:
        text = re.sub(r"\d{4}-\d{2}-\d{2}", "", report.translate(self.ARABIC_DIGITS))
        allowed = self.allowed_numbers(ctx)
        issues = []
        for n in self.NUM.findall(text):
            if n not in allowed and n.rstrip("%") not in allowed:
                issues.append(f"Ungrounded number: {n}")
        known_scores = set(self.SCORE.findall(json.dumps(ctx, default=str)))
        for s in self.SCORE.findall(text):
            if s not in known_scores and tuple(reversed(s)) not in known_scores:
                issues.append(f"Ungrounded score: {s[0]}-{s[1]}")
        for team in (ctx["match"]["home"], ctx["match"]["away"]):
            if team not in report:
                issues.append(f"Team not mentioned: {team}")
        for heading in ("## English", "## العربية"):
            if heading not in report:
                issues.append(f"Missing section: {heading}")
        english, _, arabic = report.partition("## العربية")
        for part, headings in ((english, self.REQUIRED_EN), (arabic, self.REQUIRED_AR)):
            for h in headings:  # a report cut off by the token limit loses its last sub-headings
                if not re.search(rf"^###\s*{h}", part, re.M):
                    issues.append(f"Missing sub-heading: {h}")
        return {"passed": not issues, "issues": sorted(set(issues))}


def orchestrate(ctx: dict, max_attempts: int = 3, use_ai: bool = True) -> dict:
    analyst, checker = AnalysisAgent(), ValidationAgent()
    if not use_ai:  # e.g. daily AI budget used up on the public site
        analyst.provider = None
    feedback = None
    start = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        report = analyst.run(ctx, feedback)
        verdict = checker.run(report, ctx)
        if verdict["passed"]:
            break
        feedback = verdict["issues"]
    seconds = round(time.perf_counter() - start, 1)
    if not verdict["passed"] and analyst.provider and not analyst.error:
        # never show a report that failed the check: fall back to the grounded template
        report = template_report(ctx)
        verdict = checker.run(report, ctx)
        return {"report": report, "validation": verdict, "attempts": attempt, "seconds": seconds,
                "mode": f"template (AI report failed fact check: {', '.join(feedback)})"}
    return {"report": report, "validation": verdict, "attempts": attempt, "seconds": seconds, "mode": analyst.mode}


from report_template import template_report  # noqa: E402  (offline writer, kept in its own module)
