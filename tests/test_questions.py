import json
from pathlib import Path

from ticbench.questions import build_messages, load_questions


def _all_text(messages):
    values = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            values.append(content)
        else:
            values.extend(part["text"] for part in content if part["type"] == "input_text")
    return "\n".join(values)


def test_spatial_prompt_never_contains_answer(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    record = {
        "qid": "s1",
        "answer": "SECRET_GROUND_TRUTH",
        "sequence": [
            {"type": "text", "role": "context", "content": "context"},
            {"type": "text", "role": "question", "content": "where?"},
            {"type": "text", "role": "answer", "content": "SECRET_GROUND_TRUTH"},
        ],
    }
    (folder / "dataset_caption.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_questions(folder, "spatial")[0]
    assert "SECRET_GROUND_TRUTH" not in _all_text(build_messages(folder, "spatial", loaded))


def test_visual_prompt_never_adds_assistant_answer(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "qa_pairs_1.json").write_text(json.dumps({"qa_pairs": [{"qid": "v1", "question": "choose", "answer": "LEAK"}]}), encoding="utf-8")
    (folder / "node_sequence_1.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")
    loaded = load_questions(folder, "visual")[0]
    messages = build_messages(folder, "visual", loaded)
    assert [message["role"] for message in messages] == ["user"]
    assert "LEAK" not in _all_text(messages)


def test_temporal_id_selection_does_not_read_answer(tmp_path: Path):
    folder = tmp_path / "question_1"
    folder.mkdir()
    (folder / "id_9.png").write_bytes(b"unused")
    record = {"question": "What happens?", "description": "A scene", "answer": "id_9"}
    messages = build_messages(folder, "temporal", record)
    assert "id_9" not in _all_text(messages)

