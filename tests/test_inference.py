import json
from pathlib import Path

from ticbench.inference import run_answers


def test_inference_protects_nonempty_failed_answer(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "q1", "question": "Q", "answer": "A"}]}), encoding="utf-8")
    (folder / "node_sequence_1.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    protected = [{"qid": "q1", "answer": "historical", "status": "failed"}]
    (folder / "answer_results_demo.json").write_text(json.dumps(protected), encoding="utf-8")

    def forbidden(_messages):
        raise AssertionError("non-empty historical answer must not be overwritten")

    summary = run_answers(tmp_path, "demo", forbidden, kind="visual")
    assert summary["preserved"] == 1
    assert json.loads((folder / "answer_results_demo.json").read_text(encoding="utf-8")) == protected


def test_inference_keeps_historical_qids_outside_current_dataset(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "new", "question": "Q", "answer": "A"}]}), encoding="utf-8")
    (folder / "node_sequence_1.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    old = {"qid": "old", "answer": "keep me", "status": "success"}
    (folder / "answer_results_demo.json").write_text(json.dumps([old]), encoding="utf-8")

    run_answers(tmp_path, "demo", lambda _messages: "new answer", kind="visual")
    saved = json.loads((folder / "answer_results_demo.json").read_text(encoding="utf-8"))
    assert old in saved
