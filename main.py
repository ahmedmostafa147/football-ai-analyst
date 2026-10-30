"""CLI: python main.py "Arsenal" "Liverpool" [--league EPL] [--season 2627] [--date 2026-11-01]"""
import argparse
import json
from datetime import date
from pathlib import Path

from agents import orchestrate
from retrieval import build_context, teams


def main():
    p = argparse.ArgumentParser(description="Football AI pre-match analyst")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--date", default=str(date.today()))
    p.add_argument("--league", default="EPL", choices=["EPL", "EGY"])
    p.add_argument("--season", default="2627")
    args = p.parse_args()

    known = teams(args.league, args.season)
    for t in (args.home, args.away):
        if t not in known:
            raise SystemExit(f"Unknown team '{t}'. Known teams: {', '.join(known)}")

    ctx = build_context(args.home, args.away, date.fromisoformat(args.date), args.league, args.season)
    result = orchestrate(ctx)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    stem = f"{args.date}_{args.home}_vs_{args.away}".replace(" ", "_").replace("'", "")
    (out / f"{stem}.md").write_text(result["report"], encoding="utf-8")
    (out / f"{stem}.json").write_text(
        json.dumps({"context": ctx, **result}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(result["report"])
    print(f"\nMode: {result['mode']} | attempts: {result['attempts']} | "
          f"validation: {'PASSED' if result['validation']['passed'] else 'FAILED'} {result['validation']['issues']}")
    print(f"Saved: {out / stem}.md")


if __name__ == "__main__":
    main()
