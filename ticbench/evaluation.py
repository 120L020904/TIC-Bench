from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .io import detect_dataset_kind, load_excluded_qids, question_folders, read_json, write_json_atomic
from .questions import load_questions


JudgeCaller = Callable[[list[dict[str, Any]]], str]

JUDGE_INSTRUCTIONS = """You are a strict answer evaluator. Compare the candidate with the reference using the question and optional context. Judge the candidate's final conclusion, ignoring abandoned intermediate guesses. Return only JSON with keys consistent (boolean), correct (boolean), score (number from 0 to 100), and reason (English, at most 40 words)."""


def build_judge_messages(question: str, reference: str, candidate: str, context: str = "") -> list[dict[str, Any]]:
    body = (
        f"Context:\n{context}\n\n" if context else ""
    ) + f"Question:\n{question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{candidate}"
    return [
        {"role": "developer", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": body},
    ]


def parse_judgment(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    result = json.loads(cleaned)
    if not all(key in result for key in ("consistent", "correct", "score", "reason")):
        raise ValueError("Judge response is missing required fields")
    if not isinstance(result["consistent"], bool) or not isinstance(result["correct"], bool):
        raise ValueError("Judge boolean fields are invalid")
    result["score"] = max(0.0, min(100.0, float(result["score"])))
    result["reason"] = str(result["reason"])
    return result


def aggregate_judgments(judgments: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine exactly three valid judgments using a boolean majority vote."""
    if len(judgments) != 3:
        raise ValueError("Exactly three judge results are required")
    if any(not isinstance(item.get("correct"), bool) for item in judgments.values()):
        raise ValueError("Every judge must provide a boolean correct vote")
    if any(not isinstance(item.get("consistent"), bool) for item in judgments.values()):
        raise ValueError("Every judge must provide a boolean consistent vote")

    votes = {name: item["correct"] for name, item in judgments.items()}
    consistent_votes = {name: item["consistent"] for name, item in judgments.items()}
    correct_count = sum(votes.values())
    consistent_count = sum(consistent_votes.values())
    majority = correct_count >= 2
    agreement_count = max(correct_count, 3 - correct_count)
    return {
        "correct": majority,
        "consistent": consistent_count >= 2,
        "score": round(sum(float(item["score"]) for item in judgments.values()) / 3, 4),
        "reason": f"Three-judge majority: {correct_count}/3 judges voted correct.",
        "majority_vote": majority,
        "judge_votes": votes,
        "judge_consistency_votes": consistent_votes,
        "judge_results": dict(judgments),
        "agreement_count": agreement_count,
        "unanimous": agreement_count == 3,
        "status": "complete",
    }


def _valid_judge_results(item: dict[str, Any] | None, judge_names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    raw = (item or {}).get("judge_results", {})
    if not isinstance(raw, dict):
        return {}
    return {
        name: result for name in judge_names
        if isinstance((result := raw.get(name)), dict)
        and isinstance(result.get("correct"), bool)
        and isinstance(result.get("consistent"), bool)
        and result.get("score") is not None
    }


def _candidate_map(path: Path) -> dict[str, str]:
    result = {}
    for item in read_json(path):
        if not isinstance(item, dict):
            continue
        qid = item.get("qid") or item.get("qa_id") or item.get("question_id") or item.get("id")
        answer = item.get("model_response") or item.get("answer")
        if qid is not None and answer is not None and str(answer).strip():
            result[str(qid)] = str(answer).strip()
    return result


def run_evaluation(
    root: Path,
    target: str,
    judges: Mapping[str, JudgeCaller],
    *,
    kind: str | None = None,
) -> dict[str, int]:
    root = root.resolve()
    if len(judges) != 3:
        raise ValueError("Three unique judges are required")
    judge_names = tuple(judges)
    detected = kind or detect_dataset_kind(root)
    excluded = load_excluded_qids(root)
    answer_name = f"answer_results_{target}.json"
    evaluation_name = f"evaluation_results_{target}.json"
    summary = {
        "folders": 0, "evaluated": 0, "preserved": 0,
        "judge_calls": 0, "judge_successes": 0,
        "missing_answer": 0, "failed": 0,
    }

    for folder in question_folders(root):
        answer_path = folder / answer_name
        if not answer_path.exists():
            continue
        summary["folders"] += 1
        output = folder / evaluation_name
        candidates = _candidate_map(answer_path)
        old_items = read_json(output) if output.exists() else []
        old = {str(item.get("qid")): item for item in old_items if isinstance(item, dict) and item.get("qid") is not None}
        state = dict(old)
        for record in load_questions(folder, detected):
            qid = str(record["qid"])
            if qid in excluded:
                continue
            previous = old.get(qid)
            completed = _valid_judge_results(previous, judge_names)
            if len(completed) == 3 and previous.get("status") == "complete":
                summary["preserved"] += 1
                continue
            candidate = candidates.get(qid)
            if not candidate:
                summary["missing_answer"] += 1
                continue
            base = {
                "qid": qid,
                "model_name": target,
                "question_text": record["question"],
                "status": "incomplete",
                "judge_results": completed,
            }
            messages = build_judge_messages(record["question"], record["answer"], candidate, record["description"])
            for judge_name, caller in judges.items():
                if judge_name in completed:
                    continue
                summary["judge_calls"] += 1
                try:
                    completed[judge_name] = parse_judgment(caller(messages))
                    summary["judge_successes"] += 1
                except Exception as error:
                    base["last_error"] = f"{judge_name}: {type(error).__name__}: {error}"
                    summary["failed"] += 1
                base["judge_results"] = completed
                state[qid] = base
                write_json_atomic(output, list(state.values()))

            if len(completed) == 3:
                base.update(aggregate_judgments(completed))
                base.pop("last_error", None)
                summary["evaluated"] += 1
            state[qid] = base
            write_json_atomic(output, list(state.values()))
        if state:
            write_json_atomic(output, list(state.values()))
    return summary
