from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


QUESTION_DIR_RE = re.compile(r"^question_(\d+)$")


def question_folders(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and QUESTION_DIR_RE.match(path.name)),
        key=lambda path: int(QUESTION_DIR_RE.match(path.name).group(1)),
    )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def extract_qid(item: dict[str, Any], index: int | None = None) -> str | None:
    for key in ("qid", "qa_id", "question_id", "id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"qa_{index}" if index is not None else None


def first_matching(path: Path, patterns: Iterable[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_excluded_qids(root: Path) -> set[str]:
    names = (
        "excluded_qids_final.json",
        "extended_excluded_qids.json",
        "final_qids_balanced.json",
        "all_5models_common_correct.json",
        "linear_high_acc_removed.json",
    )
    path = first_matching(root, names)
    if path is None:
        return set()
    data = read_json(path)
    if isinstance(data, list):
        return {
            qid for index, item in enumerate(data) if isinstance(item, dict)
            and (qid := extract_qid(item, index))
        }
    if isinstance(data, dict):
        for key in ("excluded_qids", "removed_qids", "excluded_easy_qids"):
            if isinstance(data.get(key), list):
                return {str(value) for value in data[key]}
        if isinstance(data.get("final_qids"), list):
            final = {str(value) for value in data["final_qids"]}
            return set() if not final else {qid for qid in _all_dataset_qids(root) if qid not in final}
    return set()


def _all_dataset_qids(root: Path) -> set[str]:
    from .questions import load_questions

    result: set[str] = set()
    for folder in question_folders(root):
        for item in load_questions(folder, detect_dataset_kind(root, folder)):
            result.add(item["qid"])
    return result


def detect_dataset_kind(root: Path, folder: Path | None = None) -> str:
    folder = folder or next(iter(question_folders(root)), root)
    if (folder / "dataset_caption.jsonl").exists():
        return "spatial"
    qa_path = first_matching(folder, ("qa_pairs*.json",))
    if qa_path:
        data = read_json(qa_path)
        if isinstance(data, dict) and isinstance(data.get("qa"), list):
            return "temporal"
        return "visual"
    raise ValueError(f"Cannot detect dataset kind under {root}")


def safe_model_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"

