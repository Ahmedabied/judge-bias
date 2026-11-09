import pytest

from judge_bias import public
from judge_bias.core import analyse


def rec(q, a, b, w, judge, turn=1):
    return {"question_id": q, "model_a": a, "model_b": b, "winner": w, "judge": judge, "turn": turn}


def test_rows_pool_humans_name_the_model_judge_and_keep_ties():
    rows, models, judges = public.to_rows([
        rec(81, "y", "x", "model_a", "gpt4_pair"),
        rec(81, "y", "x", "tie (inconsistent)", "gpt4_pair", 2),
        rec(81, "x", "y", "tie (bothbad)", "expert_3"),
        rec(81, "x", "y", "model_b", "author_1"),
    ])
    assert models == ["x", "y"] and judges == ["gpt-4", "humans"]
    assert [r["pair"] for r in rows] == [["x", "y"]] * 4  # one orientation, whatever the file says
    assert [r["winner"] for r in rows] == ["y", "tie", "tie", "y"]
    assert {r["judge"] for r in rows[2:]} == {"humans"}


def test_ties_count_half():
    recs = []
    for q in range(40):
        # gpt-4 vs two others; humans call every gpt-4 pair a tie, gpt-4 picks itself
        for other in ("a", "b"):
            recs += [rec(q, "gpt-4", other, "model_a", "gpt4_pair"), rec(q, "gpt-4", other, "tie", "expert_1")]
        w = "model_a" if q % 2 else "model_b"  # a neutral pair both judges agree on
        recs += [rec(q, "a", "b", w, "gpt4_pair"), rec(q, "a", "b", w, "expert_2")]
    rows, models, judges = public.to_rows(recs)
    res = analyse(rows, models, judges, rounds=200)
    s = res["self_preference"]["gpt-4"]
    assert s["picks_itself"] == 1.0 and s["reference_picks_it"] == 0.5
    assert s["raw"] == 0.5 and s["follows_panel"] == 1.0 and s["adjusted"] == 0.5
    assert res["judges_detail"]["humans"]["ties"] == 80


def test_fetch_refuses_a_changed_file(tmp_path, monkeypatch):
    import io

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(public.urllib.request, "urlopen", lambda *a, **k: R(b"not the file"))
    with pytest.raises(ValueError, match="does not match"):
        public.fetch(tmp_path)
    assert not list(tmp_path.iterdir())
