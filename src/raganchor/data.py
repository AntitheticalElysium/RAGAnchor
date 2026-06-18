"""RAGTruth loader. Three task types, three shapes of `source_info`:
  - Summary  -> str  (one document)
  - Data2txt -> dict (structured business record)
  - QA       -> dict {question, passages}  (passages are "passage N:" delimited)
The train/test split lives on the *responses*, not the sources, so source
selection goes through `load_responses`. Responses also carry RAGTruth's human
hallucination spans (`labels`) — kept here for validating our own judge later."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from raganchor.config import RAGTRUTH_DIR

_PASSAGE_PREFIX = re.compile(r"^passage\s*\d+\s*:\s*", re.IGNORECASE)


@dataclass
class Source:
    source_id: str
    task_type: str  # "QA" | "Summary" | "Data2txt"
    context: str  # grounding text: the doc / rendered record / joined passages
    question: str | None = None  # QA only
    passages: list[str] = field(default_factory=list)  # QA: units to retrieve over


@dataclass
class Response:
    source_id: str
    model: str
    split: str
    response: str
    quality: str
    labels: list[dict]  # human hallucination spans; [] = clean


def _render_data2txt(record: dict) -> str:
    """Flatten the structured business record into readable key: value lines."""
    lines: list[str] = []
    for key, val in record.items():
        if val is None or val == "":
            continue
        if isinstance(val, dict):
            inner = "; ".join(f"{k}: {v}" for k, v in val.items() if v not in (None, ""))
            if inner:
                lines.append(f"{key}: {inner}")
        else:
            lines.append(f"{key}: {val}")
    return "\n".join(lines)


def _split_passages(passages: str) -> list[str]:
    chunks = [c.strip() for c in passages.split("\n\n") if c.strip()]
    return [_PASSAGE_PREFIX.sub("", c).strip() for c in chunks]


def _to_source(row: dict) -> Source:
    task = row["task_type"]
    info = row["source_info"]
    if task == "QA":
        passages = _split_passages(info["passages"])
        return Source(
            source_id=row["source_id"],
            task_type=task,
            context="\n\n".join(passages),
            question=info["question"],
            passages=passages,
        )
    if task == "Data2txt":
        context = _render_data2txt(info)
        return Source(row["source_id"], task, context, passages=[context])
    # Summary: source_info is the raw document string
    return Source(row["source_id"], task, info.strip(), passages=[info.strip()])


def load_sources(
    task_types: list[str] | None = None, path: Path | None = None
) -> dict[str, Source]:
    path = path or RAGTRUTH_DIR / "source_info.jsonl"
    out: dict[str, Source] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if task_types and row["task_type"] not in task_types:
                continue
            out[row["source_id"]] = _to_source(row)
    return out


def load_responses(
    split: str | None = None, path: Path | None = None
) -> Iterator[Response]:
    path = path or RAGTRUTH_DIR / "response.jsonl"
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if split and row.get("split") != split:
                continue
            yield Response(
                source_id=row["source_id"],
                model=row["model"],
                split=row["split"],
                response=row["response"],
                quality=row.get("quality", ""),
                labels=row.get("labels", []),
            )
