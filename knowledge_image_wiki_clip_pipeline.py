#!/usr/bin/env python3
"""
CLIP-backed continuation of the local knowledge image wiki prototype.

This keeps the same local document -> keyword card -> graph -> AI walk flow,
but upgrades the retrieval layer:

  node coordinate = CLIP image embedding of the generated keyword card
  query coordinate = CLIP text embedding of the imagined visual target

If the local CLIP model is unavailable, it falls back to the TF-IDF path from
knowledge_image_wiki_demo.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import knowledge_image_wiki_demo as demo


DEFAULT_QUERY = (
    "Build an AI knowledge wiki that uploads notes, turns knowledge into keyword "
    "images, and lets an AI investigate only the relevant visual nodes while working."
)


IMAGINED_STEPS = [
    "uploaded notes splitting into small keyword image cards arranged on a desk",
    "an image card acting as a doorway or address for associative memory",
    "a coordinate map where image embeddings and text embeddings meet in one space",
    "an AI walker carrying a small light and reading only nearby useful cards",
    "an evidence board with selected source cards connected into an answer trail",
    "a safety tag attached to every generated image saying source text remains the ground truth",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CLIP-backed knowledge image wiki from markdown notes."
    )
    parser.add_argument("--docs", type=Path, default=demo.ROOT / "demo_knowledge")
    parser.add_argument("--out", type=Path, default=demo.ROOT / "generated_clip_wiki")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--backend", choices=["auto", "clip", "tfidf"], default="auto")
    parser.add_argument("--model", default="sentence-transformers/clip-ViT-B-32")
    return parser.parse_args()


def configure_paths(docs: Path, out: Path) -> None:
    demo.DOC_DIR = docs.resolve()
    demo.OUT_DIR = out.resolve()
    demo.IMG_DIR = demo.OUT_DIR / "keyword_images"
    demo.VAULT_DIR = demo.OUT_DIR / "my-wiki"


def load_clip_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_clip_images(model: Any, nodes: list[demo.Node]) -> np.ndarray:
    images = [Image.open(node.image_path).convert("RGB") for node in nodes]
    return np.asarray(model.encode(images, normalize_embeddings=True), dtype=np.float32)


def encode_clip_text(model: Any, phrases: list[str]) -> np.ndarray:
    return np.asarray(model.encode(phrases, normalize_embeddings=True), dtype=np.float32)


def clip_walk(
    nodes: list[demo.Node],
    graph,
    model: Any,
    query: str,
    imagined_steps: list[str],
) -> list[dict]:
    node_matrix = encode_clip_images(model, nodes)
    query_phrases = [f"{query}. Visual target: {step}" for step in imagined_steps]
    query_matrix = encode_clip_text(model, query_phrases)
    sim = cosine_similarity(query_matrix, node_matrix)

    visited: set[str] = set()
    walk: list[dict] = []
    for step_idx, phrase in enumerate(imagined_steps, start=1):
        ranked = np.argsort(sim[step_idx - 1])[::-1]
        chosen = None
        for idx in ranked:
            if nodes[idx].id not in visited:
                chosen = int(idx)
                break
        if chosen is None:
            chosen = int(ranked[0])
        node = nodes[chosen]
        visited.add(node.id)
        neighbors = sorted(
            graph.neighbors(node.id),
            key=lambda nid: graph.edges[node.id, nid].get("weight", 0),
            reverse=True,
        )[:2]
        walk.append(
            {
                "step": step_idx,
                "imagined": phrase,
                "node_id": node.id,
                "title": node.title,
                "score": float(sim[step_idx - 1, chosen]),
                "keywords": node.keywords,
                "source": str(node.path.relative_to(demo.ROOT))
                if node.path.is_relative_to(demo.ROOT)
                else str(node.path),
                "summary": demo.summarize(node.text),
                "neighbors": neighbors,
            }
        )
    return walk


def write_report(
    nodes: list[demo.Node],
    graph,
    walk: list[dict],
    html_path: Path,
    canvas_path: Path,
    notes_path: Path,
    answer_path: Path,
    backend: str,
    query: str,
) -> Path:
    report = {
        "prototype": "clip-backed-knowledge-image-wiki",
        "backend": backend,
        "query": query,
        "documents": len(nodes),
        "images": len(nodes),
        "edges": graph.number_of_edges(),
        "html": str(html_path),
        "canvas": str(canvas_path),
        "obsidian_walk_note": str(notes_path),
        "answer": str(answer_path),
        "walk": walk,
        "production_upgrade_path": [
            "Replace deterministic card images with GPT-Image-2 or another image generator.",
            "Keep source text attached to every visual node.",
            "Use CLIP image embeddings for generated cards and CLIP text/image embeddings for imagined targets.",
            "Add a task planner that decides when to stop walking and synthesize an answer.",
        ],
    }
    path = demo.OUT_DIR / "clip_run_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_obsidian_notes(nodes: list[demo.Node], walk: list[dict], query: str) -> Path:
    notes_dir = demo.VAULT_DIR / "notes"
    sources_dir = demo.VAULT_DIR / "sources"
    notes_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    note_by_node: dict[str, str] = {}
    for node in nodes:
        source_copy = sources_dir / node.path.name
        shutil.copy2(node.path, source_copy)
        note_name = f"{node.id}-{demo.slug(node.title)}.md"
        note_by_node[node.id] = note_name
        body = f"""# {node.title}

![[keyword_images/{node.image_path.name}]]

## Keywords
{", ".join(node.keywords)}

## Source
[[sources/{source_copy.name}]]

## Summary
{demo.summarize(node.text, 520)}

## Retrieval Role
This note is a visual entry point. The image card is used for navigation; the source text remains the ground truth.
"""
        (notes_dir / note_name).write_text(body, encoding="utf-8")

    lines = [
        "# AI 조사 경로",
        "",
        f"질문: {query}",
        "",
        "이 노트는 CLIP text cue가 가장 가까운 keyword image node로 착지한 경로입니다.",
        "",
    ]
    for item in walk:
        note_name = note_by_node[item["node_id"]]
        lines.extend(
            [
                f"## {item['step']}. [[notes/{note_name}|{item['title']}]]",
                "",
                f"- 떠올린 시각 단서: {item['imagined']}",
                f"- 매칭 점수: {item['score']:.2f}",
                f"- 키워드: {', '.join(item['keywords'][:5])}",
                f"- 요약: {item['summary']}",
                "",
            ]
        )

    path = demo.VAULT_DIR / "AI 조사 경로.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_answer(walk: list[dict], query: str) -> Path:
    path = demo.OUT_DIR / "answer.md"
    evidence = "\n".join(
        f"- Step {item['step']}: {item['title']} ({item['score']:.2f}) — {item['summary']}"
        for item in walk
    )
    answer = f"""# Synthesized Answer

## Question
{query}

## Answer
The structure is viable as a task-oriented image wiki:

1. Uploaded knowledge is chunked into source-grounded cards.
2. Each card receives a keyword image that acts as a visual address.
3. CLIP places those image cards into a shared coordinate space.
4. During work, the AI forms an imagined visual cue from the task and enters the nearest image node.
5. The AI reads only the source text attached to the visited nodes, then synthesizes an answer from that evidence.
6. Generated images must remain indexes, not facts; every card keeps its original source text attached.

## Evidence Path
{evidence}

## Current Limitation
This run uses deterministic local keyword cards, not GPT-Image-2. The retrieval layer is already CLIP-backed, so replacing the card generator with GPT-Image-2 is now a contained swap rather than a redesign.
"""
    path.write_text(answer, encoding="utf-8")
    return path


def export_readme(backend: str, html_path: Path, canvas_path: Path, notes_path: Path, answer_path: Path) -> Path:
    path = demo.OUT_DIR / "README.md"
    body = f"""# CLIP Knowledge Image Wiki

Generated by `knowledge_image_wiki_clip_pipeline.py`.

- Backend: `{backend}`
- HTML viewer: `{html_path.name}`
- Obsidian canvas: `{canvas_path.relative_to(demo.OUT_DIR)}`
- Obsidian walk note: `{notes_path.relative_to(demo.OUT_DIR)}`
- Synthesized answer: `{answer_path.name}`

Run again:

```bash
python3 knowledge_image_wiki_clip_pipeline.py --backend clip --docs demo_knowledge --out generated_clip_wiki
```

Use a different markdown folder:

```bash
python3 knowledge_image_wiki_clip_pipeline.py --backend auto --docs /path/to/notes --out /path/to/output
```
"""
    path.write_text(body, encoding="utf-8")
    return path


def annotate_html_backend(html_path: Path, backend: str) -> None:
    text = html_path.read_text(encoding="utf-8")
    text = text.replace(
        "API 없이 만든 로컬 미니어처입니다. 문서를 넣고, 키워드를 뽑고, 키워드 이미지 카드를 만들고, 관련 노드를 그래프로 정렬한 뒤, 작업 질의에서 떠오른 시각적 단서가 필요한 지식 노드로 진입하는 흐름을 검증합니다.",
        f"API 없이 만든 로컬 미니어처입니다. 문서를 넣고, 키워드를 뽑고, 키워드 이미지 카드를 만들고, CLIP 좌표계로 관련 노드를 그래프로 정렬한 뒤, 작업 질의에서 떠오른 시각적 단서가 필요한 지식 노드로 진입하는 흐름을 검증합니다. Retrieval backend: {backend}.",
    )
    text = text.replace(
        "4. 그래프 정렬: vector similarity + spring layout",
        "4. 그래프 정렬: CLIP image embeddings + spring layout",
    )
    text = text.replace(
        "5. 작업 중 조사: imagined visual cue → nearest node → source text",
        "5. 작업 중 조사: CLIP text cue → nearest image node → source text",
    )
    html_path.write_text(text, encoding="utf-8")


def run_tfidf(query: str) -> tuple[str, list[demo.Node], Any, list[dict]]:
    nodes = demo.build_nodes()
    _, matrix, _ = demo.vectorize(nodes)
    graph, _ = demo.graph_from_vectors(nodes, matrix)
    walk = demo.imagine_and_walk(nodes, matrix, graph)
    return "tfidf", nodes, graph, walk


def run_clip(model_name: str, query: str) -> tuple[str, list[demo.Node], Any, list[dict]]:
    nodes = demo.build_nodes()
    model = load_clip_model(model_name)
    node_matrix = encode_clip_images(model, nodes)
    graph, _ = demo.graph_from_vectors(nodes, node_matrix)
    walk = clip_walk(nodes, graph, model, query, IMAGINED_STEPS)
    return f"clip:{model_name}", nodes, graph, walk


def main() -> None:
    args = parse_args()
    configure_paths(args.docs, args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.backend == "tfidf":
        backend, nodes, graph, walk = run_tfidf(args.query)
    else:
        try:
            backend, nodes, graph, walk = run_clip(args.model, args.query)
        except Exception as exc:
            if args.backend == "clip":
                raise
            print(f"CLIP unavailable, falling back to TF-IDF: {type(exc).__name__}: {exc}")
            backend, nodes, graph, walk = run_tfidf(args.query)

    canvas_path = demo.export_canvas(nodes, graph, walk)
    html_path = demo.export_html(nodes, graph, walk)
    annotate_html_backend(html_path, backend)
    notes_path = export_obsidian_notes(nodes, walk, args.query)
    answer_path = export_answer(walk, args.query)
    readme_path = export_readme(backend, html_path, canvas_path, notes_path, answer_path)
    report_path = write_report(nodes, graph, walk, html_path, canvas_path, notes_path, answer_path, backend, args.query)

    print(f"backend={backend}")
    print(f"documents={len(nodes)}")
    print(f"keyword_images={len(nodes)}")
    print(f"edges={graph.number_of_edges()}")
    print(f"walk_steps={len(walk)}")
    print(f"html={html_path}")
    print(f"canvas={canvas_path}")
    print(f"obsidian_walk_note={notes_path}")
    print(f"answer={answer_path}")
    print(f"readme={readme_path}")
    print(f"report={report_path}")
    for item in walk:
        print(f"{item['step']}. {item['imagined']} -> {item['title']} ({item['score']:.2f})")


if __name__ == "__main__":
    main()
