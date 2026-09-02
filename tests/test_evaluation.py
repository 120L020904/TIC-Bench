import json
from pathlib import Path

from ticbench.evaluation import aggregate_judgments, parse_judgment, run_evaluation


def test_parse_judgment_clamps_score():
    result = parse_judgment('```json\n{"consistent":true,"correct":true,"score":120,"reason":"ok"}\n```')
    assert result["score"] == 100.0


def test_three_judge_majority_and_agreement():
    judgments = {
        "j1": {"consistent": True, "correct": True, "score": 100, "reason": "yes"},
        "j2": {"consistent": True, "correct": True, "score": 80, "reason": "yes"},
        "j3": {"consistent": False, "correct": False, "score": 10, "reason": "no"},
    }
    result = aggregate_judgments(judgments)
    assert result["correct"] is True
    assert result["judge_votes"] == {"j1": True, "j2": True, "j3": False}
    assert result["agreement_count"] == 2
    assert result["unanimous"] is False


def test_evaluation_resumes_valid_results(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "q1", "question": "Q", "answer": "A"}]}), encoding="utf-8")
    (folder / "answer_results_demo.json").write_text(json.dumps([{"qid": "q1", "answer": "A", "status": "success"}]), encoding="utf-8")
    one = {"score": 100, "correct": True, "consistent": True, "reason": "same"}
    existing = [{
        "qid": "q1", "score": 100, "correct": True, "consistent": True,
        "reason": "same", "status": "complete",
        "judge_results": {"j1": one, "j2": one, "j3": one},
    }]
    (folder / "evaluation_results_demo.json").write_text(json.dumps(existing), encoding="utf-8")

    def forbidden(_messages):
        raise AssertionError("valid evaluation should be preserved")

    summary = run_evaluation(
        tmp_path, "demo", {"j1": forbidden, "j2": forbidden, "j3": forbidden}, kind="visual"
    )
    assert summary["preserved"] == 1


def test_evaluation_calls_three_judges(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "q1", "question": "Q", "answer": "A"}]}), encoding="utf-8")
    (folder / "answer_results_demo.json").write_text(json.dumps([{"qid": "q1", "answer": "A"}]), encoding="utf-8")
    raw = '{"consistent":true,"correct":true,"score":100,"reason":"same"}'
    summary = run_evaluation(
        tmp_path, "demo", {"j1": lambda _: raw, "j2": lambda _: raw, "j3": lambda _: raw}, kind="visual"
    )
    saved = json.loads((folder / "evaluation_results_demo.json").read_text(encoding="utf-8"))[0]
    assert summary["judge_calls"] == 3
    assert saved["majority_vote"] is True
    assert saved["unanimous"] is True


def test_evaluation_resumes_only_missing_judge(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "q1", "question": "Q", "answer": "A"}]}), encoding="utf-8")
    (folder / "answer_results_demo.json").write_text(json.dumps([{"qid": "q1", "answer": "A"}]), encoding="utf-8")
    raw = '{"consistent":true,"correct":true,"score":100,"reason":"same"}'

    def fail(_):
        raise RuntimeError("temporary")

    first = run_evaluation(
        tmp_path, "demo", {"j1": lambda _: raw, "j2": lambda _: raw, "j3": fail}, kind="visual"
    )
    assert first["judge_calls"] == 3
    assert first["judge_successes"] == 2

    def forbidden(_):
        raise AssertionError("completed judge must not run again")

    second = run_evaluation(
        tmp_path, "demo", {"j1": forbidden, "j2": forbidden, "j3": lambda _: raw}, kind="visual"
    )
    assert second["judge_calls"] == 1
    saved = json.loads((folder / "evaluation_results_demo.json").read_text(encoding="utf-8"))[0]
    assert saved["status"] == "complete"
