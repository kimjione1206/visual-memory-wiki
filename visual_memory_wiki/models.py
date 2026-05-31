from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    id: str
    title: str
    source_path: Path
    text: str
    keywords: list[str]
    prompt: str
    image_path: Path | None = None
    x: float = 0.0
    y: float = 0.0


@dataclass
class WalkStep:
    step: int
    cue: str
    node_id: str
    title: str
    score: float
    keywords: list[str]
    source: str
    summary: str


@dataclass
class RetrievalRow:
    step: int
    cue: str
    expected_id: str | None
    expected_title: str | None
    top_id: str
    top_title: str
    top_score: float
    expected_rank: int | None = None
    correct: bool | None = None
    top3: list[dict] = field(default_factory=list)


def slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9가-힣]+", "-", text.strip().lower()).strip("-")
    return text[:56] or "node"


def summarize(text: str, limit: int = 260) -> str:
    clean = " ".join(text.split())
    return clean[:limit].rstrip() + ("..." if len(clean) > limit else "")
