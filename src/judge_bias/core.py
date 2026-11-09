"""Does an LLM judge prefer answers written by its own model?

Several answer models answer the same questions. Judge models then compare every pair of answers
twice, once in each order, so a preference for the first slot cancels out. For a model that is both
an answerer and a judge, self-preference is how often it picks its own answer minus how often the
judges with no stake in that pair pick it, corrected for how closely it follows those judges on
pairs where it has no stake itself. Zero means no self-preference. A bootstrap over the questions
gives the interval.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

PROMPTS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
PROMPTS_SHA256 = "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7"

JUDGE_SYSTEM = (
    "You are a careful, impartial judge. You compare two answers to the same question and decide which one is "
    "better: more helpful, more correct and clearer. The order of the answers, their length and their style must "
    "not decide it."
)
JUDGE_TEMPLATE = (
    "[Question]\n{question}\n\n[Answer A]\n{a}\n\n[Answer B]\n{b}\n\n"
    "Which answer is better? Reply with exactly one letter: A or B."
)
# The reply must lead with the letter, optionally after "Answer" or "The better answer is", and the letter must
# stand alone ("A", "B.", "**A**", "Answer: B", "A is better"). "A better answer is B" is unreadable, not A.
VERDICT = re.compile(r"^\W*(?:(?i:(?:the\s+)?(?:better\s+)?answer(?:\s+is)?)\W*)?([AB])(?=\s*$|[.,:;!)\]*\n]|\s+is\b)")


class Model(Protocol):
    name: str

    def generate(self, prompt: str, system: str | None = None, max_tokens: int = 512) -> str: ...


@dataclass(frozen=True)
class Question:
    qid: str
    category: str
    text: str


def fetch_prompts(dest: str | Path) -> Path:
    """MT-Bench's 80 questions (Apache-2.0, from the FastChat repository), pinned by hash."""
    dest = Path(dest)
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == PROMPTS_SHA256:
        return dest
    with urllib.request.urlopen(PROMPTS_URL, timeout=60) as r:
        body = r.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != PROMPTS_SHA256:
        raise ValueError(f"prompt file hash {digest} does not match the pinned {PROMPTS_SHA256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(body)
    return dest


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            out.append(Question(str(q["question_id"]), str(q.get("category", "")), str(q["turns"][0])))
    return out


def parse_verdict(text: str) -> str | None:
    m = VERDICT.match(text.strip())
    return m.group(1) if m else None


class Cache:
    """Answers and verdicts are appended to JSON lines files as they come, so a stopped run resumes."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        self.rows[row["key"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]


def run(questions: list[Question], answerers: list[Model], judges: list[Model], out: str | Path,
        max_answer_tokens: int = 400, log=print) -> dict:
    """Collect answers and verdicts (resumable), then analyse. Returns the results dict."""
    names = [m.name for m in answerers]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("give at least two different answer models")
    out = Path(out)
    answers = Cache(out / "answers.jsonl")
    verdicts = Cache(out / "verdicts.jsonl")
    for m in answerers:
        for q in questions:
            key = f"{m.name}|{q.qid}"
            if answers.get(key) is None:
                answers.put({"key": key, "model": m.name, "qid": q.qid, "text": m.generate(q.text, None, max_answer_tokens)})
        log(f"answers from {m.name}: done")
    for j in judges:
        for q in questions:
            texts = {n: answers.get(f"{n}|{q.qid}")["text"] for n in names}
            for m, n in pairs(names):
                for order, (a, b) in enumerate([(m, n), (n, m)]):
                    key = f"{j.name}|{q.qid}|{m}|{n}|{order}"
                    if verdicts.get(key) is not None:
                        continue
                    raw = j.generate(JUDGE_TEMPLATE.format(question=q.text, a=texts[a], b=texts[b]), JUDGE_SYSTEM, 8)
                    v = parse_verdict(raw)
                    verdicts.put({"key": key, "judge": j.name, "qid": q.qid, "pair": [m, n], "order": order,
                                  "slot_a": a, "slot_b": b, "raw": raw[:40], "verdict": v,
                                  "winner": {"A": a, "B": b}.get(v) if v else None})
        log(f"verdicts from {j.name}: done")
    return analyse(list(verdicts.rows.values()), names, [j.name for j in judges])


def _cells(rows: list[dict]) -> dict[tuple, list[float]]:
    """(judge, question, turn, m, n) -> the judge's readable verdicts on that pair: 1 picked m, 0 picked n, and a tie
    (in imported judgments) counts half for each side. Turn is None for local runs (first turns only); imported
    judgments compare each turn with the same turn."""
    acc: dict[tuple, list[float]] = {}
    for r in rows:
        w = r["winner"]
        if w is not None:
            m, n = r["pair"]
            acc.setdefault((r["judge"], r["qid"], r.get("turn"), m, n), []).append(0.5 if w == "tie" else 1.0 if w == m else 0.0)
    return acc


def _pct(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    vals = sorted(vals)
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[min(len(vals) - 1, int(0.975 * len(vals)))], 4)]


def analyse(rows: list[dict], answerers: list[str], judges: list[str], rounds: int = 2000, seed: int = 0) -> dict:
    """Self-preference per model that is both an answerer and a judge.

    The yardstick is the panel: judges that answer nothing, so they have no stake in any pair. (Without a
    panel, the judges with no stake in each pair are used, and only the raw number is given.)

    raw: how often X picks its own answer minus how often the panel picks it.

    A noisy judge drifts toward 50%, which looks like self-preference whenever its own answers are weaker.
    So, on the pairs where X has no stake, fit how X's verdicts follow the panel's, through the middle:
    p - 0.5 = k (r - 0.5). k is 1 for a judge that agrees with the panel, near 0 for a coin flip. With no
    self-preference X should pick itself at 0.5 + k (r - 0.5) on its own pairs, so
    adjusted: mean(own pick rate - 0.5 - k (r - 0.5)). It needs three or more answer models and a panel.

    The panel's pick rate on a pair is itself an estimate (often one human verdict), and noise in it flattens the
    fitted k toward 0, which would read a judge that simply knows its answer is better as self-preferring. So the
    noise is measured, as the spread between verdicts on the same pair (pooled over pairs with two or more), and
    taken out of the fit: k = sum (p - 0.5)(r - 0.5) / (sum (r - 0.5)^2 - summed noise variance of r).

    Intervals: questions resampled with replacement, everything recomputed each time, 95% percentile.
    """
    cells = _cells(rows)
    rate = {key: statistics.fmean(v) for key, v in cells.items()}
    panel = [j for j in judges if j not in answerers]
    ss = df = 0.0
    for (j, *_), v in cells.items():
        if j in panel and len(v) > 1:
            mu = statistics.fmean(v)
            ss += sum((x - mu) ** 2 for x in v)
            df += len(v) - 1
    noise = ss / df if df else 0.0  # the variance of one panel verdict around its pair's mean
    result: dict = {"answerers": answerers, "judges": judges, "panel": panel,
                    "judges_detail": {}, "self_preference": {}}
    for j in judges:
        mine = [r for r in rows if r["judge"] == j]
        valid = [r for r in mine if r["winner"] is not None]
        lettered = [r for r in valid if r.get("verdict") in ("A", "B")]  # imported judgments have no slots
        both: dict[tuple, list[str]] = {}
        for r in lettered:
            if r.get("order") is not None:
                both.setdefault((r["qid"], tuple(r["pair"])), []).append(r["winner"])
        two = [w for w in both.values() if len(w) == 2]
        result["judges_detail"][j] = {
            "verdicts": len(mine),
            "unreadable": len(mine) - len(valid),
            "ties": sum(1 for r in valid if r["winner"] == "tie"),
            "first_slot_rate": round(sum(1 for r in lettered if r["verdict"] == "A") / len(lettered), 4) if lettered else None,
            "consistent_rate": round(sum(1 for w in two if w[0] == w[1]) / len(two), 4) if two else None,
        }

    def reference(x: str, key: tuple) -> tuple[float, float] | None:
        """The panel's pick rate for m on (question, turn, m, n), and the noise variance of that rate."""
        q, t, m, n = key
        refs = [o for o in (panel or [o for o in judges if o not in (x, m, n)]) if (o, *key) in rate]
        if not refs:
            return None
        var = noise * sum(1 / len(cells[(o, *key)]) for o in refs) / len(refs) ** 2 if panel else 0.0
        return statistics.fmean(rate[(o, *key)] for o in refs), var

    items = sorted({key[1:] for key in cells}, key=lambda k: (k[0], k[1] or 0, k[2], k[3]))
    for x in answerers:
        if x not in judges:
            continue
        own: dict[str, list[tuple[float, float]]] = {}             # q -> (X's pick rate for X, reference for X)
        neutral: dict[str, list[tuple[float, float, float]]] = {}  # q -> (X's rate for m, reference for m, its noise)
        for key in items:
            q, _, m, n = key
            p, ref = rate.get((x, *key)), reference(x, key)
            if p is None or ref is None:
                continue
            r, var = ref
            if x == m:
                own.setdefault(q, []).append((p, r))
            elif x == n:
                own.setdefault(q, []).append((1 - p, 1 - r))
            elif panel:
                neutral.setdefault(q, []).append((p, r, var))

        def fit(sample: list[str]) -> tuple[float | None, float | None]:
            """k through the middle with the panel's noise taken out, and the share of the spread that is real."""
            nt = [pt for q in sample for pt in neutral.get(q, [])]
            sxx = sum((r - 0.5) ** 2 for _, r, _ in nt)
            spread = sxx - sum(v for _, _, v in nt)
            if sxx <= 0 or spread <= 0:
                return None, None
            return sum((p - 0.5) * (r - 0.5) for p, r, _ in nt) / spread, spread / sxx

        def stats(sample: list[str]) -> tuple[float | None, float | None]:
            o = [pt for q in sample for pt in own.get(q, [])]
            if not o:
                return None, None
            raw = statistics.fmean(a - b for a, b in o)
            k, _ = fit(sample)
            if k is None:
                return raw, None
            return raw, statistics.fmean(a - 0.5 - k * (b - 0.5) for a, b in o)

        oq = sorted(own)
        if not oq:
            continue
        rng = random.Random(seed)
        boot_raw, boot_adj = [], []
        for _ in range(rounds):
            raw, adj = stats([oq[rng.randrange(len(oq))] for _ in oq])
            if raw is not None:
                boot_raw.append(raw)
            if adj is not None:
                boot_adj.append(adj)
        pts = [pt for q in oq for pt in own[q]]
        raw, adj = stats(oq)
        k, reliability = fit(oq)
        result["self_preference"][x] = {
            "comparisons": len(pts),
            "picks_itself": round(statistics.fmean(a for a, _ in pts), 4),
            "reference_picks_it": round(statistics.fmean(b for _, b in pts), 4),
            "raw": round(raw, 4),
            "raw_ci95": _pct(boot_raw),
            "follows_panel": round(k, 4) if k is not None else None,
            "panel_reliability": round(reliability, 4) if reliability is not None else None,
            "adjusted": round(adj, 4) if adj is not None else None,
            "adjusted_ci95": _pct(boot_adj) if adj is not None else None,
        }
    return result


def _ci(v) -> str:
    return f"{v[0]:+.3f} to {v[1]:+.3f}" if v else "n/a"


def _num(v, signed: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def markdown(result: dict, meta: dict | None = None) -> str:
    lines = ["# Results", ""]
    if meta:
        lines += [f"- {k}: {v}" for k, v in meta.items()] + [""]
    lines += ["## Self-preference", "",
              "| model | comparisons | picks itself | panel picks it | raw | 95% interval | follows the panel (k) | adjusted | 95% interval | panel reliability |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in result["self_preference"].items():
        lines.append(f"| {m} | {s['comparisons']} | {_num(s['picks_itself'])} | {_num(s['reference_picks_it'])} | "
                     f"{_num(s['raw'], True)} | {_ci(s['raw_ci95'])} | {_num(s['follows_panel'])} | "
                     f"{_num(s['adjusted'], True)} | {_ci(s['adjusted_ci95'])} | {_num(s.get('panel_reliability'))} |")
    lines += ["", "## Judges", "", "| judge | verdicts | unreadable | ties | picks the first slot | same winner in both orders |",
              "|---|---|---|---|---|---|"]
    for j, d in result["judges_detail"].items():
        lines.append(f"| {j} | {d['verdicts']} | {d['unreadable']} | {d['ties']} | {_num(d['first_slot_rate'])} | "
                     f"{_num(d['consistent_rate'])} |")
    return "\n".join(lines) + "\n"


def questions_subset(questions: Iterable[Question], limit: int | None) -> list[Question]:
    qs = list(questions)
    return qs[:limit] if limit else qs
