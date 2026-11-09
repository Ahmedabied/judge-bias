"""Published judgments, analysed with the same method: no model needed.

MT-Bench human judgments (LMSYS, CC-BY-4.0): GPT-4's pairwise verdicts and human experts' verdicts on the
answers of six models to the 80 MT-Bench questions, both turns. GPT-4 is one of the answer models and one
of the judges; the humans answer nothing, so they are the panel.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

DATASET = "lmsys/mt_bench_human_judgments"
REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
FILES = {
    "data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet": "57068327ffb5f8fddb15fedbbfe40b26fe862da2326331e5cf64a42320a00e71",
    "data/human-00000-of-00001-25f4910818759289.parquet": "4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769",
}
JUDGE_NAMES = {"gpt4_pair": "gpt-4"}  # the judge is named after the answer model it is


def fetch(dest: str | Path) -> list[Path]:
    dest = Path(dest)
    out = []
    for name, sha in FILES.items():
        path = dest / Path(name).name
        if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha):
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            got = hashlib.sha256(body).hexdigest()
            if got != sha:
                raise ValueError(f"{name}: hash {got} does not match the pinned {sha}")
            dest.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        out.append(path)
    return out


def to_rows(records: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    """Dataset records to verdict rows. Every human judge is pooled into one panel judge, 'humans'."""
    models = sorted({r["model_a"] for r in records} | {r["model_b"] for r in records})
    rows = []
    for r in records:
        judge = JUDGE_NAMES.get(r["judge"], "humans")
        a, b = r["model_a"], r["model_b"]
        m, n = sorted((a, b), key=models.index)
        w = r["winner"]
        winner = a if w == "model_a" else b if w == "model_b" else "tie" if str(w).startswith("tie") else None
        rows.append({"judge": judge, "qid": str(r["question_id"]), "pair": [m, n], "turn": r["turn"],
                     "verdict": None, "order": None, "winner": winner, "label": w, "source": r["judge"]})
    judges = sorted({row["judge"] for row in rows}, key=lambda j: (j == "humans", j))
    return rows, models, judges


def load(dest: str | Path) -> tuple[list[dict], list[str], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise OSError("reading the published judgments needs pyarrow: pip install 'judge-bias[public]'") from e
    cols = ["question_id", "model_a", "model_b", "winner", "judge", "turn"]
    records = [rec for p in fetch(dest) for rec in pq.read_table(p, columns=cols).to_pylist()]
    return to_rows(records)
