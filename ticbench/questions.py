from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Any

from .io import extract_qid, first_matching, read_json


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def text(value: str) -> dict[str, str]:
    return {"type": "input_text", "text": value}


def image(path: Path) -> dict[str, str]:
    return {"type": "input_image", "image_url": image_data_url(path)}


def load_questions(folder: Path, kind: str) -> list[dict[str, Any]]:
    if kind == "spatial":
        records = []
        with (folder / "dataset_caption.jsonl").open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                item = json.loads(line)
                records.append({
                    "qid": extract_qid(item, index),
                    "question": next((str(part.get("content", "")) for part in item.get("sequence", []) if part.get("role") == "question"), ""),
                    "answer": str(item.get("answer") or next((part.get("content", "") for part in item.get("sequence", []) if part.get("role") == "answer"), "")),
                    "description": "\n\n".join(str(part.get("content", "")) for part in item.get("sequence", []) if part.get("role") not in {"question", "answer"}),
                    "source": item,
                })
        return records

    qa_path = first_matching(folder, ("qa_pairs*.json",))
    if qa_path is None:
        return []
    data = read_json(qa_path)
    key = "qa" if kind == "temporal" else "qa_pairs"
    description = str(data.get("description") or data.get("descipion") or "") if isinstance(data, dict) else ""
    items = data.get(key, []) if isinstance(data, dict) else data
    return [
        {
            "qid": extract_qid(item, index),
            "question": str(item.get("question", "")),
            "answer": str(item.get("answer", "")),
            "description": description,
            "source": item,
        }
        for index, item in enumerate(items)
        if isinstance(item, dict)
    ]


def build_messages(folder: Path, kind: str, record: dict[str, Any], mode: str = "full") -> list[dict[str, Any]]:
    if kind == "spatial":
        content = _spatial_content(folder, record["source"], mode)
        return [{"role": "user", "content": content}]
    if kind == "visual":
        content = _visual_content(folder, record["question"], mode)
        return [{"role": "user", "content": content}]
    if kind == "temporal":
        content = _temporal_content(folder, record["description"], record["question"], mode)
        return [
            {"role": "developer", "content": "Analyze the visual story. Refer to each character, animal, or person only by its introduced id_X identifier."},
            {"role": "user", "content": content},
        ]
    raise ValueError(f"Unsupported dataset kind: {kind}")


def _spatial_content(folder: Path, item: dict[str, Any], mode: str) -> list[dict[str, str]]:
    if mode not in {"full", "text_only", "images_only", "question_images_only"}:
        raise ValueError(f"Unsupported spatial mode: {mode}")
    content: list[dict[str, str]] = []
    replaced = {str(value) for value in item.get("metadata", {}).get("text_replaced_crop_ids", [])}
    crops = sorted((folder / "crops").glob("*.jpg")) + sorted((folder / "crops").glob("*.png"))
    if mode in {"full", "images_only"}:
        for path in crops:
            if path.stem not in replaced:
                content.extend((image(path), text(f"This image is {path.stem}.")))
    elif mode == "text_only":
        content.extend(text(f"Image label: {path.stem}.") for path in crops if path.stem not in replaced)

    for part in item.get("sequence", []):
        role, part_type, value = part.get("role"), part.get("type"), str(part.get("content", ""))
        if role == "answer":
            continue
        if part_type == "image" and mode in {"full", "images_only", "question_images_only"}:
            path = folder / "annotated" / Path(value).name
            if path.is_file():
                content.append(image(path))
        elif role == "question":
            content.append(text(value))
        elif part_type == "text" and mode in {"full", "text_only"}:
            content.append(text(value))
    return content


def _visual_content(folder: Path, question: str, mode: str) -> list[dict[str, str]]:
    if mode not in {"full", "text_only", "images_only", "question_images_only"}:
        raise ValueError(f"Unsupported visual mode: {mode}")
    sequence_path = first_matching(folder, ("node_sequence*.json",))
    sequence = read_json(sequence_path) if sequence_path else {}
    nodes = sequence.get("nodes", sequence if isinstance(sequence, list) else [])
    entries: list[tuple[str, Path]] = []
    for node in nodes:
        node_id = str(node.get("node_id"))
        path = folder / f"node_{node_id}.png"
        if path.is_file():
            entries.append((f"image {node_id}", path))
    for object_id in dict.fromkeys(re.findall(r"(?:object_|obj_|<object_|\[object_)(\d+)", question)):
        path = folder / "objects" / f"{int(object_id)}.png"
        if path.is_file():
            entries.append((f"object_{int(object_id)}", path))
    content: list[dict[str, str]] = []
    for label, path in entries:
        if mode in {"full", "text_only"}:
            content.append(text(f"Reference image: {label}"))
        if mode in {"full", "images_only", "question_images_only"}:
            content.append(image(path))
    content.append(text(question))
    return content


def _temporal_content(folder: Path, description: str, question: str, mode: str) -> list[dict[str, str]]:
    if mode not in {"full", "text_only", "images_only", "question_images_only"}:
        raise ValueError(f"Unsupported temporal mode: {mode}")
    content: list[dict[str, str]] = []
    mentioned = sorted({int(value) for value in re.findall(r"\bid_(\d+)\b", f"{description}\n{question}")})
    if mode in {"full", "images_only"}:
        for identifier in mentioned:
            path = folder / f"id_{identifier}.png"
            if path.is_file():
                content.extend((image(path), text(f"The image above is id_{identifier}.")))
    elif mode == "text_only":
        content.extend(text(f"Character identifier: id_{identifier}.") for identifier in mentioned)

    scenes = sorted(folder.glob("scene_*.png"), key=lambda path: int(re.search(r"(\d+)", path.stem).group(1)))
    if mode in {"full", "images_only", "question_images_only"}:
        for path in scenes:
            content.extend((image(path), text(f"[{path.stem}]")))
    if mode in {"full", "text_only"} and description:
        content.append(text(f"Story Description:\n{description}"))
    content.append(text(f"Question:\n{question}"))
    return content

