import hashlib
import io
import json

import pytest

from judge_bias import core
from judge_bias.backends import Fake, Ollama
from judge_bias.cli import main
from judge_bias.core import Question, analyse, parse_verdict, run

QS = [Question(str(i), "t", f"question {i}") for i in range(60)]


def experiment(tmp_path, answerers, panel=(), rounds=300):
    judges = list(answerers) + list(panel)
    run(QS, answerers, judges, tmp_path, log=lambda s: None)
    rows = [json.loads(l) for l in (tmp_path / "verdicts.jsonl").read_text().splitlines()]
    return analyse(rows, [m.name for m in answerers], [j.name for j in judges], rounds=rounds), judges


def fair_panel():
    return [Fake("p1", "fair"), Fake("p2", "fair")]


def test_fair_judges_show_no_self_preference(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "fair", 1), Fake("b", "fair", 1), Fake("c", "fair", 2)], fair_panel())
    for m in "abc":
        s = res["self_preference"][m]
        assert s["raw"] == 0 and s["adjusted"] == 0, (m, s)


def test_self_biased_judge_is_found(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "self", 1), Fake("b", "fair", 1), Fake("c", "fair", 2)], fair_panel())
    a = res["self_preference"]["a"]
    assert a["picks_itself"] == 1.0
    assert a["adjusted"] > 0.5 and a["adjusted_ci95"][0] > 0.4
    assert res["self_preference"]["b"]["adjusted"] == 0


def test_noise_alone_is_not_called_self_preference(tmp_path):
    # b is a coin flip and writes weaker answers than c. Raw numbers blame it; the adjustment does not.
    res, _ = experiment(tmp_path, [Fake("a", "fair", 1), Fake("b", "noisy", 0), Fake("c", "fair", 2)], fair_panel())
    b = res["self_preference"]["b"]
    assert b["raw"] > 0.2 and b["raw_ci95"][0] > 0.1
    assert abs(b["follows_panel"]) < 0.3
    assert b["adjusted_ci95"][0] < 0 < b["adjusted_ci95"][1]


def test_position_bias_cancels_and_is_reported(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "first", 1), Fake("b", "fair", 1), Fake("c", "fair", 2)], fair_panel())
    d = res["judges_detail"]["a"]
    assert d["first_slot_rate"] == 1.0 and d["consistent_rate"] == 0.0
    s = res["self_preference"]["a"]
    assert s["picks_itself"] == 0.5  # both orders: always slot A means each answer wins once
    assert s["follows_panel"] == 0.0 and s["adjusted"] == 0.0


def test_unreadable_verdicts_are_counted_not_guessed(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "mute", 1), Fake("b", "fair", 1), Fake("c", "fair", 2)], fair_panel())
    d = res["judges_detail"]["a"]
    assert d["unreadable"] == d["verdicts"] == 60 * 3 * 2
    assert d["first_slot_rate"] is None
    assert "a" not in res["self_preference"]


def test_without_panel_only_raw_is_given(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "self", 1), Fake("b", "fair", 1), Fake("c", "fair", 2)])
    a = res["self_preference"]["a"]
    assert res["panel"] == []
    assert a["raw"] > 0.4 and a["adjusted"] is None and a["follows_panel"] is None


def test_two_answerers_work_without_the_adjustment(tmp_path):
    res, _ = experiment(tmp_path, [Fake("a", "self", 1), Fake("b", "fair", 2)], fair_panel())
    assert res["self_preference"]["a"]["raw"] > 0.4
    assert res["self_preference"]["a"]["adjusted"] is None


def test_run_resumes_without_calling_models_again(tmp_path):
    ans = [Fake("a", "fair", 1), Fake("b", "fair", 2), Fake("c", "fair", 3)]
    _, judges = experiment(tmp_path, ans, fair_panel())
    before = {j.name: j.calls for j in judges}
    run(QS, ans, judges, tmp_path, log=lambda s: None)
    assert {j.name: j.calls for j in judges} == before
    lines = (tmp_path / "verdicts.jsonl").read_text().splitlines()
    assert len(lines) == len({json.loads(l)["key"] for l in lines}) == 5 * 60 * 3 * 2


def test_run_refuses_one_answerer(tmp_path):
    with pytest.raises(ValueError):
        run(QS, [Fake("a")], [Fake("a")], tmp_path)
    with pytest.raises(ValueError):
        run(QS, [Fake("a"), Fake("a")], [Fake("a")], tmp_path)


@pytest.mark.parametrize("text,want", [
    ("A", "A"), ("B", "B"), (" B.", "B"), ("**A**", "A"), ("Answer: B", "B"), ("Answer B", "B"),
    ("The better answer is A.", "A"), ("A is better", "A"), ("B\nbecause it is shorter", "B"), ("[A]", "A"),
    ("A better answer is B", None), ("I cannot decide.", None), ("", None), ("AB", None), ("Both", None),
    ("a", None),
])
def test_parse_verdict(text, want):
    assert parse_verdict(text) == want


def test_fetch_checks_the_pinned_hash(tmp_path, monkeypatch):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(core.urllib.request, "urlopen", lambda *a, **k: R(b"tampered"))
    with pytest.raises(ValueError, match="does not match"):
        core.fetch_prompts(tmp_path / "q.jsonl")
    assert not (tmp_path / "q.jsonl").exists()

    body = b'{"question_id": 81, "category": "writing", "turns": ["Write a haiku.", "Again."]}\n'
    monkeypatch.setattr(core, "PROMPTS_SHA256", hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(core.urllib.request, "urlopen", lambda *a, **k: R(body))
    path = core.fetch_prompts(tmp_path / "q.jsonl")
    monkeypatch.setattr(core.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used for a cached file"))
    assert core.fetch_prompts(tmp_path / "q.jsonl") == path
    assert core.load_questions(path) == [Question("81", "writing", "Write a haiku.")]


def test_ollama_backend_is_local_only():
    with pytest.raises(ValueError):
        Ollama("m", "http://example.com:11434")
    assert Ollama("m", "http://127.0.0.1:11535/").host == "http://127.0.0.1:11535"


def test_ollama_request_is_greedy_and_seeded(monkeypatch):
    seen = {}

    def fake_post(self, path, body):
        seen.update(path=path, body=body)
        return {"response": "B"}

    monkeypatch.setattr(Ollama, "_post", fake_post)
    assert Ollama("m").generate("q", "sys", 8) == "B"
    assert seen["path"] == "/api/generate"
    assert seen["body"]["options"] == {"temperature": 0, "seed": 0, "num_predict": 8}
    assert seen["body"]["system"] == "sys" and seen["body"]["stream"] is False


def test_demo_and_report_cli(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| alpha |" in out and "follows the panel" in out and "panel reliability" in out
    assert main(["report", str(tmp_path), "--answerers", "alpha,bravo,charlie"]) == 0
    assert "| bravo |" in capsys.readouterr().out
