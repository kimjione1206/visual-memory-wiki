from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from visual_memory_wiki.models import Node, RetrievalRow, WalkStep, summarize


DEFAULT_CUES = [
    "uploaded notes splitting into small keyword image cards arranged on a desk",
    "an image card acting as a doorway or address for associative memory",
    "a coordinate map where image embeddings and text embeddings meet in one space",
    "an AI walker carrying a small light and reading only nearby useful cards",
    "an evidence board with selected source cards connected into an answer trail",
    "a safety tag attached to every generated image saying source text remains the ground truth",
]


def rank_cues(nodes: list[Node], cues: list[str], cue_matrix: np.ndarray, node_matrix: np.ndarray, expected: dict[int, str] | None = None) -> dict:
    scores = cosine_similarity(cue_matrix, node_matrix)
    rows: list[RetrievalRow] = []
    reciprocal_ranks: list[float] = []
    hits = 0
    node_ids = [node.id for node in nodes]
    for step_idx in range(scores.shape[0]):
        ranked = np.argsort(scores[step_idx])[::-1]
        top_idx = int(ranked[0])
        top_node = nodes[top_idx]
        expected_id = expected.get(step_idx + 1) if expected else None
        expected_title = None
        expected_rank = None
        correct = None
        if expected_id:
            expected_node = nodes[node_ids.index(expected_id)]
            expected_title = expected_node.title
            rank_ids = [node_ids[int(i)] for i in ranked]
            expected_rank = rank_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / expected_rank)
            correct = top_node.id == expected_id
            hits += int(correct)
        rows.append(
            RetrievalRow(
                step=step_idx + 1,
                cue=cues[step_idx],
                expected_id=expected_id,
                expected_title=expected_title,
                top_id=top_node.id,
                top_title=top_node.title,
                top_score=float(scores[step_idx, top_idx]),
                expected_rank=expected_rank,
                correct=correct,
                top3=[
                    {
                        "id": nodes[int(i)].id,
                        "title": nodes[int(i)].title,
                        "score": float(scores[step_idx, int(i)]),
                    }
                    for i in ranked[:3]
                ],
            )
        )
    accuracy = hits / len(rows) if expected else None
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None
    return {"accuracy": accuracy, "mrr": mrr, "rows": rows}


def make_walk(nodes: list[Node], cues: list[str], cue_matrix: np.ndarray, node_matrix: np.ndarray) -> list[WalkStep]:
    scores = cosine_similarity(cue_matrix, node_matrix)
    visited: set[str] = set()
    walk: list[WalkStep] = []
    for i, cue in enumerate(cues):
        ranked = np.argsort(scores[i])[::-1]
        chosen = int(ranked[0])
        for idx in ranked:
            if nodes[int(idx)].id not in visited:
                chosen = int(idx)
                break
        node = nodes[chosen]
        visited.add(node.id)
        walk.append(
            WalkStep(
                step=i + 1,
                cue=cue,
                node_id=node.id,
                title=node.title,
                score=float(scores[i, chosen]),
                keywords=node.keywords,
                source=str(node.source_path),
                summary=summarize(node.text),
            )
        )
    return walk
