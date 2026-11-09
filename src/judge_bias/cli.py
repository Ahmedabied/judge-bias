"""Command line: judge-bias fetch | run | report"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .backends import Fake, Ollama
from .core import Question, analyse, fetch_prompts, load_questions, markdown, questions_subset, run


def cmd_fetch(a) -> int:
    print(fetch_prompts(a.dest))
    return 0


def cmd_run(a) -> int:
    answerers = [Ollama(m, a.host) for m in a.answerers.split(",")]
    judges = [Ollama(m, a.host) for m in a.judges.split(",")] if a.judges else []
    questions = questions_subset(load_questions(fetch_prompts(a.prompts)), a.limit)
    out = Path(a.out)
    t0 = time.time()
    result = run(questions, answerers, judges, out, a.max_answer_tokens, log=lambda s: print(s, flush=True))
    if not judges:
        print(f"answers saved in {out / 'answers.jsonl'}; run again with --judges for the verdicts")
        return 0
    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "questions": f"{len(questions)} MT-Bench first turns (Apache-2.0, FastChat repository)",
        "decoding": "greedy, temperature 0, seed 0",
        "models": ", ".join(f"{m.name} ({m.digest()[:12]})" for m in {m.name: m for m in answerers + judges}.values()),
        "wall time": f"{(time.time() - t0) / 60:.1f} min",
    }
    (out / "results.json").write_text(json.dumps({"meta": meta, **result}, indent=2), encoding="utf-8")
    (out / "results.md").write_text(markdown(result, meta), encoding="utf-8")
    print(markdown(result, meta))
    return 0


def cmd_demo(a) -> int:
    """Scripted models, no download: one judge prefers itself, one is only noisy, the rest are fair."""
    questions = [Question(str(i), "demo", f"Demo question {i}") for i in range(80)]
    answerers = [Fake("alpha", "self", 1), Fake("bravo", "noisy", 1), Fake("charlie", "fair", 2)]
    judges = answerers + [Fake("delta", "fair", 0), Fake("echo", "fair", 0)]
    out = Path(a.out)
    for f in ("answers.jsonl", "verdicts.jsonl"):
        (out / f).unlink(missing_ok=True)
    print(markdown(run(questions, answerers, judges, out, log=lambda s: None),
                   {"what": "alpha favours its own answers, bravo is only noisy, charlie is fair; delta and echo "
                            "answer nothing and form the panel. alpha and bravo write weaker answers than charlie."}))
    return 0


def cmd_report(a) -> int:
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "verdicts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    answerers = a.answerers.split(",")
    judges = sorted({r["judge"] for r in rows})
    print(markdown(analyse(rows, answerers, judges)))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="judge-bias", description="Does an LLM judge prefer its own model's answers?")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the MT-Bench questions (hash-checked)")
    f.add_argument("--dest", default="data/mt_bench_questions.jsonl")
    f.set_defaults(fn=cmd_fetch)
    r = sub.add_parser("run", help="answers, then verdicts in both orders, then the analysis (resumable)")
    r.add_argument("--answerers", required=True, help="answer models, comma separated (three or more for the noise control)")
    r.add_argument("--judges", help="judge models, comma separated: the answerers plus panel models that answer "
                                   "nothing. Leave out to collect answers only")
    r.add_argument("--host", default="http://127.0.0.1:11434")
    r.add_argument("--prompts", default="data/mt_bench_questions.jsonl")
    r.add_argument("--limit", type=int)
    r.add_argument("--max-answer-tokens", type=int, default=400)
    r.add_argument("--out", default="results/run")
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("demo", help="run the whole analysis on scripted models, no download")
    d.add_argument("--out", default="results/demo")
    d.set_defaults(fn=cmd_demo)
    rep = sub.add_parser("report", help="re-analyse saved verdicts")
    rep.add_argument("out")
    rep.add_argument("--answerers", required=True)
    rep.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except (OSError, ValueError) as e:
        print(f"judge-bias: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
