from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import question_folders, read_json


def build_report(root: Path) -> dict[str, Any]:
    models: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for folder in question_folders(root.resolve()):
        for path in folder.glob("evaluation_results_*.json"):
            model = path.stem.removeprefix("evaluation_results_")
            data = read_json(path)
            models[model].extend(item for item in data if isinstance(item, dict))
    report = {}
    for model, items in sorted(models.items()):
        scored = [item for item in items if item.get("score") is not None]
        correct = sum(bool(item.get("correct")) for item in scored)
        report[model] = {
            "evaluated": len(scored),
            "correct": correct,
            "accuracy": round(correct / len(scored), 6) if scored else None,
            "mean_score": round(sum(float(item["score"]) for item in scored) / len(scored), 4) if scored else None,
            "unanimous": sum(item.get("unanimous") is True for item in scored),
            "unanimous_rate": round(sum(item.get("unanimous") is True for item in scored) / len(scored), 6) if scored else None,
        }
    return report
