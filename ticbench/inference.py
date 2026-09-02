from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import detect_dataset_kind, load_excluded_qids, question_folders, read_json, safe_model_key, write_json_atomic
from .questions import build_messages, load_questions


ModelCaller = Callable[[list[dict[str, Any]]], str]


def run_answers(
    root: Path,
    model_name: str,
    caller: ModelCaller | None,
    *,
    kind: str | None = None,
    mode: str = "full",
    dry_run: bool = False,
) -> dict[str, int]:
    """Generate missing answers while preserving every existing non-empty answer."""
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    detected = kind or detect_dataset_kind(root)
    excluded = load_excluded_qids(root)
    filename = f"answer_results_{safe_model_key(model_name)}.json"
    summary = {"folders": 0, "generated": 0, "preserved": 0, "excluded": 0, "failed": 0, "previewed": 0}

    for folder in question_folders(root):
        records = load_questions(folder, detected)
        if not records:
            continue
        summary["folders"] += 1
        output = folder / filename
        existing_items = read_json(output) if output.exists() else []
        existing = {
            str(item.get("qid") or item.get("qa_id") or item.get("id")): item
            for item in existing_items if isinstance(item, dict)
        }
        results: list[dict[str, Any]] = []
        current_qids: set[str] = set()
        changed = False
        for record in records:
            qid = str(record["qid"])
            current_qids.add(qid)
            old = existing.get(qid)
            old_answer = (old or {}).get("model_response") or (old or {}).get("answer")
            if old_answer is not None and str(old_answer).strip():
                results.append(old)
                summary["preserved"] += 1
                continue
            if qid in excluded:
                results.append({"qid": qid, "status": "excluded", "model_response": None})
                summary["excluded"] += 1
                changed = True
                continue
            messages = build_messages(folder, detected, record, mode)
            if dry_run:
                summary["previewed"] += 1
                continue
            if caller is None:
                raise ValueError("caller is required unless dry_run is enabled")
            try:
                answer = caller(messages)
                status = "success" if answer and answer.strip() else "failed"
                results.append({
                    "qid": qid,
                    "question": record["question"],
                    "ground_truth": record["answer"],
                    "model_response": answer or None,
                    "status": status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                summary["generated" if status == "success" else "failed"] += 1
            except Exception as error:
                results.append({
                    "qid": qid,
                    "question": record["question"],
                    "ground_truth": record["answer"],
                    "model_response": None,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                summary["failed"] += 1
            changed = True
            write_json_atomic(output, results)
        # Preserve out-of-scope historical records as well. A changed dataset
        # must never cause an old, valid answer to disappear from its result file.
        results.extend(
            item for item in existing_items
            if isinstance(item, dict)
            and str(item.get("qid") or item.get("qa_id") or item.get("id")) not in current_qids
        )
        if changed and not dry_run:
            write_json_atomic(output, results)
    return summary
