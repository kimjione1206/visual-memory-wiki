from __future__ import annotations

import math

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from visual_memory_wiki.models import Node


def build_similarity_graph(nodes: list[Node], matrix: np.ndarray, threshold: float = 0.015, top_k: int = 3) -> tuple[nx.Graph, np.ndarray]:
    sim = cosine_similarity(matrix)
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id, node=node)
    for i, node in enumerate(nodes):
        ranked = np.argsort(sim[i])[::-1]
        added = 0
        for j in ranked:
            if i == j or sim[i, j] < threshold:
                continue
            graph.add_edge(node.id, nodes[int(j)].id, weight=float(sim[i, j]))
            added += 1
            if added >= top_k:
                break
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.9, iterations=200)
    for node in nodes:
        node.x = float(pos[node.id][0])
        node.y = float(pos[node.id][1])
    return graph, sim


def scale_positions(nodes: list[Node], width: int = 1200, height: int = 760) -> dict[str, tuple[float, float]]:
    xs = [node.x for node in nodes]
    ys = [node.y for node in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 90

    def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
        if math.isclose(lo, hi):
            return (out_lo + out_hi) / 2
        return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)

    return {
        node.id: (
            scale(node.x, min_x, max_x, pad, width - pad),
            scale(node.y, min_y, max_y, pad, height - pad),
        )
        for node in nodes
    }
