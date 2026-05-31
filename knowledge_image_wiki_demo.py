#!/usr/bin/env python3
"""
Local prototype for a knowledge -> keyword image -> navigable wiki pipeline.

No paid API is required. This intentionally uses local TF-IDF vectors and
deterministic visual cards as a small-scale stand-in for:
  GPT/Image model keyword cards + CLIP image embeddings.
"""

from __future__ import annotations

import html
import json
import math
import re
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DOC_DIR = ROOT / "demo_knowledge"
OUT_DIR = ROOT / "generated_wiki"
IMG_DIR = OUT_DIR / "keyword_images"
VAULT_DIR = OUT_DIR / "my-wiki"


SAMPLE_DOCS = {
    "01_image_as_link.md": """# Image as link

The wiki treats an image as an address, not decoration. A node is entered by
visual resemblance: grids, fins, rivers, stars, maps, and hardware textures can
all become routes. The goal is to replace a hand-authored hyperlink with a
memory-like visual cue.

Keywords: image link, visual cue, associative route, memory anchor, visual wiki
""",
    "02_upload_to_keywords.md": """# Knowledge upload to keyword images

Uploaded notes are split into chunks. Each chunk is summarized, and a few
keywords become compact image prompts. The image is a stable handle for the
knowledge. A human remembers the card visually; an AI can retrieve the same
card by moving through embedding space.

Keywords: upload, chunk, keyword extraction, image prompt, knowledge card
""",
    "03_clip_space.md": """# Shared embedding space

CLIP-like systems place text and images into a shared coordinate system. A text
description such as a branching river delta can land near an image of circuit
traces, because both contain visual branching. This is useful when the search
query is an imagined image rather than a literal term.

Keywords: CLIP, embedding space, image vector, text vector, visual search
""",
    "04_ai_walker.md": """# AI walker

The walker does not read the entire vault. It starts from the current task,
imagines a visual target, enters the closest stored image node, reads that node,
then follows nearby nodes only if they help the work. The path becomes a
traceable chain of visual reasoning.

Keywords: AI walker, task context, imagined image, partial reading, reasoning path
""",
    "05_obsidian_canvas.md": """# Obsidian canvas integration

JSON Canvas can store image cards, text cards, and edges. Network layout places
related image nodes near one another. The canvas is useful because the result
is not trapped in a demo; it becomes part of a permanent vault that can be
dragged, zoomed, and edited.

Keywords: Obsidian, JSON Canvas, graph layout, image node, permanent vault
""",
    "06_hardware_visuals.md": """# Hardware visual metaphors

Hardware images are visually rich: heat-sink fins resemble city skylines,
circuit traces resemble rivers, solder pads resemble constellations, and chip
dies resemble aerial maps. This makes hardware a good sandbox for testing
visual association.

Keywords: hardware, heat sink, PCB trace, constellation, chip die
""",
    "07_task_oriented_research.md": """# Task-oriented research

The valuable version is not random wandering. The system should answer a work
question by visiting only the cards that are useful for the answer. It should
show what it saw, why it moved, and which evidence nodes shaped the response.

Keywords: research task, selective retrieval, evidence, answer synthesis, audit trail
""",
    "08_limits.md": """# Limits

Generated images are useful memory handles, but they can hallucinate details.
They should index the knowledge, not replace the source text. The safest design
keeps the original chunk attached to every visual card and uses the image only
as a navigational surface.

Keywords: limitation, hallucination, source text, grounding, safe design
""",
}


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "into",
    "only",
    "they",
    "should",
    "become",
    "because",
    "rather",
    "than",
    "not",
    "can",
    "useful",
    "uses",
    "used",
    "from",
    "will",
    "what",
    "why",
    "which",
    "when",
    "where",
    "how",
    "are",
    "is",
    "as",
    "a",
    "an",
    "to",
    "of",
    "in",
    "by",
    "it",
    "or",
    "on",
    "all",
    "few",
}


PALETTES = [
    ("#17324D", "#E8F1F2", "#F2A541"),
    ("#31493C", "#F5F1E3", "#D65A31"),
    ("#463F3A", "#F4F3EE", "#6CA6C1"),
    ("#2E4057", "#F6F5AE", "#D1495B"),
    ("#1B4965", "#CAE9FF", "#5FA8D3"),
    ("#2B2D42", "#EDF2F4", "#EF233C"),
    ("#355070", "#EAAC8B", "#B56576"),
    ("#283618", "#FEFAE0", "#BC6C25"),
]


@dataclass
class Node:
    id: str
    title: str
    path: Path
    text: str
    keywords: list[str]
    prompt: str
    image_path: Path
    cluster: str
    x: float = 0.0
    y: float = 0.0


def ensure_sample_docs() -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    if any(DOC_DIR.glob("*.md")):
        return
    for name, body in SAMPLE_DOCS.items():
        (DOC_DIR / name).write_text(body, encoding="utf-8")


def clean_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def extract_keywords(docs: list[str], top_k: int = 5) -> list[list[str]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        ngram_range=(1, 2),
        max_features=700,
    )
    matrix = vectorizer.fit_transform(docs)
    features = np.array(vectorizer.get_feature_names_out())
    result: list[list[str]] = []
    for row in matrix:
        scores = row.toarray()[0]
        ranked = scores.argsort()[::-1]
        terms: list[str] = []
        for idx in ranked:
            if scores[idx] <= 0:
                break
            term = features[idx]
            if any(term in existing or existing in term for existing in terms):
                continue
            terms.append(term)
            if len(terms) == top_k:
                break
        result.append(terms)
    return result


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_motif(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], kind: str, fg: str, accent: str) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx = x0 + w // 2
    cy = y0 + h // 2
    if kind == "graph":
        pts = [(x0 + 36, y0 + 42), (x0 + 118, y0 + 28), (x0 + 202, y0 + 74), (x0 + 84, y0 + 140), (x0 + 184, y0 + 152)]
        for a, b in [(0, 1), (1, 2), (1, 3), (3, 4), (2, 4)]:
            draw.line([pts[a], pts[b]], fill=fg, width=5)
        for px, py in pts:
            draw.ellipse((px - 13, py - 13, px + 13, py + 13), fill=accent)
    elif kind == "image":
        for i in range(5):
            xx = x0 + 22 + i * 40
            draw.rounded_rectangle((xx, y0 + 28, xx + 34, y1 - 30), radius=4, outline=fg, width=4)
        draw.arc((x0 + 60, y0 + 50, x1 - 60, y1 - 50), 0, 360, fill=accent, width=8)
    elif kind == "hardware":
        for i in range(13):
            xx = x0 + 18 + i * 17
            draw.line((xx, y0 + 22, xx, y1 - 22), fill=fg, width=8)
        draw.line((x0 + 18, cy, x1 - 18, cy), fill=accent, width=10)
    elif kind == "memory":
        for r in range(22, 92, 18):
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=fg if r % 36 else accent, width=5)
        draw.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), fill=accent)
    elif kind == "research":
        draw.rectangle((x0 + 36, y0 + 32, x1 - 36, y1 - 32), outline=fg, width=5)
        for i in range(5):
            y = y0 + 54 + i * 24
            draw.line((x0 + 58, y, x1 - 58, y), fill=accent if i == 2 else fg, width=5)
        draw.ellipse((x1 - 90, y1 - 90, x1 - 42, y1 - 42), outline=accent, width=6)
        draw.line((x1 - 54, y1 - 54, x1 - 22, y1 - 22), fill=accent, width=7)
    elif kind == "safety":
        draw.polygon([(cx, y0 + 28), (x1 - 42, cy), (cx, y1 - 28), (x0 + 42, cy)], outline=fg, width=6)
        draw.line((cx, y0 + 70, cx, y1 - 76), fill=accent, width=8)
        draw.ellipse((cx - 5, y1 - 62, cx + 5, y1 - 52), fill=accent)
    else:
        for i in range(6):
            offset = i * 18
            draw.arc((x0 + 24 + offset, y0 + 28, x1 - 24 - offset, y1 - 28), 200, 340, fill=fg, width=5)
        draw.line((x0 + 40, y1 - 48, x1 - 40, y0 + 48), fill=accent, width=7)


def motif_for(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ["clip", "embedding", "vector", "space"]):
        return "graph"
    if any(k in lowered for k in ["image", "visual", "keyword", "prompt"]):
        return "image"
    if any(k in lowered for k in ["hardware", "heat", "pcb", "chip", "circuit"]):
        return "hardware"
    if any(k in lowered for k in ["walker", "task", "research", "evidence"]):
        return "research"
    if any(k in lowered for k in ["limit", "safe", "source", "ground"]):
        return "safety"
    if any(k in lowered for k in ["memory", "anchor", "wiki"]):
        return "memory"
    return "route"


def generate_card(node_id: str, title: str, keywords: list[str], prompt: str) -> Path:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    palette = PALETTES[int(re.sub(r"\D", "", node_id) or "0") % len(PALETTES)]
    bg, fg, accent = palette
    image = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    body_font = font(22)
    small_font = font(18)

    draw.rounded_rectangle((22, 22, 490, 490), radius=22, outline=fg, width=4)
    draw_motif(draw, (56, 72, 456, 258), motif_for(prompt), fg, accent)

    title_lines = wrap_text(draw, title, title_font, 410)[:2]
    y = 292
    for line in title_lines:
        draw.text((52, y), line, font=title_font, fill=fg)
        y += 40
    draw.line((52, y + 6, 460, y + 6), fill=accent, width=4)
    y += 28
    kw = " / ".join(keywords[:4])
    for line in wrap_text(draw, kw, body_font, 408)[:3]:
        draw.text((52, y), line, font=body_font, fill=fg)
        y += 30
    draw.text((52, 458), "keyword image card", font=small_font, fill=accent)

    path = IMG_DIR / f"{node_id}_{slug(title)}.png"
    image.save(path)
    return path


def slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9가-힣]+", "-", text.strip().lower()).strip("-")
    return text[:48] or "node"


def build_nodes() -> list[Node]:
    ensure_sample_docs()
    files = sorted(DOC_DIR.glob("*.md"))
    raw_texts = [path.read_text(encoding="utf-8") for path in files]
    clean_texts = [clean_markdown(text) for text in raw_texts]
    keyword_sets = extract_keywords(clean_texts, top_k=6)

    nodes: list[Node] = []
    for idx, (path, raw, clean, keywords) in enumerate(zip(files, raw_texts, clean_texts, keyword_sets), start=1):
        title = title_from_markdown(raw, path.stem)
        prompt = f"{title}. Visual metaphor: {', '.join(keywords[:5])}. {clean[:240]}"
        node_id = f"n{idx:02d}"
        image_path = generate_card(node_id, title, keywords, prompt)
        cluster = motif_for(prompt)
        nodes.append(
            Node(
                id=node_id,
                title=title,
                path=path,
                text=clean,
                keywords=keywords,
                prompt=prompt,
                image_path=image_path,
                cluster=cluster,
            )
        )
    return nodes


def vectorize(nodes: list[Node], extra_queries: Iterable[str] = ()):
    corpus = [f"{n.title}\n{' '.join(n.keywords)}\n{n.prompt}\n{n.text}" for n in nodes]
    corpus.extend(extra_queries)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        ngram_range=(1, 2),
        max_features=1000,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix[: len(nodes)], matrix[len(nodes) :]


def graph_from_vectors(nodes: list[Node], matrix) -> tuple[nx.Graph, np.ndarray]:
    sim = cosine_similarity(matrix)
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id, node=node)
    for i, node in enumerate(nodes):
        ranked = np.argsort(sim[i])[::-1]
        added = 0
        for j in ranked:
            if i == j:
                continue
            if sim[i, j] < 0.015:
                continue
            graph.add_edge(node.id, nodes[j].id, weight=float(sim[i, j]))
            added += 1
            if added >= 3:
                break
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.9, iterations=200)
    for node in nodes:
        node.x = float(pos[node.id][0])
        node.y = float(pos[node.id][1])
    return graph, sim


def imagine_and_walk(nodes: list[Node], matrix, graph: nx.Graph) -> list[dict]:
    task = "Design an AI knowledge wiki that does not read the whole vault, but investigates only the needed image-linked knowledge while working."
    imagined = [
        "uploaded notes split into chunks, each chunk becoming a keyword extraction image prompt knowledge card",
        "a visual cue where the image itself is an address, an associative route and memory anchor",
        "a shared coordinate map where CLIP text vectors and image vectors land in the same embedding space",
        "a lone AI walker using task context and an imagined image to do partial reading instead of reading the entire vault",
        "a research task board with selective retrieval, evidence cards, answer synthesis and an audit trail",
        "a safe design guardrail: generated images may hallucinate, so source text grounding stays attached to every card",
    ]
    _, node_matrix, q_matrix = vectorize(nodes, imagined + [task])
    query_matrix = q_matrix[: len(imagined)]
    sim = cosine_similarity(query_matrix, node_matrix)

    visited: set[str] = set()
    walk: list[dict] = []
    for step_idx, phrase in enumerate(imagined, start=1):
        ranked = np.argsort(sim[step_idx - 1])[::-1]
        chosen = None
        for idx in ranked:
            if nodes[idx].id not in visited:
                chosen = idx
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
                "source": str(node.path.relative_to(ROOT)),
                "summary": summarize(node.text),
                "neighbors": neighbors,
            }
        )
    return walk


def summarize(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text[:limit].rstrip() + ("..." if len(text) > limit else "")


def scale_positions(nodes: list[Node], width: int = 1200, height: int = 760) -> dict[str, tuple[float, float]]:
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 90

    def scale(v: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
        if math.isclose(lo, hi):
            return (out_lo + out_hi) / 2
        return out_lo + (v - lo) * (out_hi - out_lo) / (hi - lo)

    return {
        n.id: (
            scale(n.x, min_x, max_x, pad, width - pad),
            scale(n.y, min_y, max_y, pad, height - pad),
        )
        for n in nodes
    }


def rel(path: Path) -> str:
    return str(path.relative_to(OUT_DIR))


def export_canvas(nodes: list[Node], graph: nx.Graph, walk: list[dict]) -> Path:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    vault_image_dir = VAULT_DIR / "keyword_images"
    vault_image_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        shutil.copy2(node.image_path, vault_image_dir / node.image_path.name)
    positions = scale_positions(nodes, width=1800, height=1200)
    walk_ids = [item["node_id"] for item in walk]
    canvas = {"nodes": [], "edges": []}
    for node in nodes:
        x, y = positions[node.id]
        canvas["nodes"].append(
            {
                "id": node.id,
                "type": "file",
                "file": str(Path("keyword_images") / node.image_path.name),
                "x": int(x),
                "y": int(y),
                "width": 220,
                "height": 220,
            }
        )
        canvas["nodes"].append(
            {
                "id": f"{node.id}_text",
                "type": "text",
                "text": f"{node.title}\n{', '.join(node.keywords[:4])}",
                "x": int(x),
                "y": int(y) + 230,
                "width": 220,
                "height": 90,
            }
        )
    for a, b, data in graph.edges(data=True):
        color = "4" if a in walk_ids and b in walk_ids else "1"
        canvas["edges"].append(
            {
                "id": f"e_{a}_{b}",
                "fromNode": a,
                "toNode": b,
                "color": color,
                "label": f"{data.get('weight', 0):.2f}",
            }
        )
    path = VAULT_DIR / "auto-knowledge-image-wiki.canvas"
    path.write_text(json.dumps(canvas, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_html(nodes: list[Node], graph: nx.Graph, walk: list[dict]) -> Path:
    positions = scale_positions(nodes)
    walk_ids = [item["node_id"] for item in walk]
    edge_lines = []
    for a, b, data in graph.edges(data=True):
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        is_walk = a in walk_ids and b in walk_ids
        edge_lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'class="edge {"walk-edge" if is_walk else ""}" stroke-width="{2 + data.get("weight", 0) * 8:.1f}" />'
        )
    node_svg = []
    for node in nodes:
        x, y = positions[node.id]
        idx = walk_ids.index(node.id) + 1 if node.id in walk_ids else None
        node_svg.append(
            f"""
            <g class="graph-node {'walk-node' if idx else ''}" transform="translate({x:.1f},{y:.1f})">
              <circle r="{34 if idx else 26}"></circle>
              <image href="{html.escape(rel(node.image_path))}" x="-24" y="-24" width="48" height="48" clip-path="circle(24px at 24px 24px)"></image>
              {'<text class="step-badge" x="27" y="-24">' + str(idx) + '</text>' if idx else ''}
              <text class="node-label" y="51">{html.escape(node.title)}</text>
            </g>
            """
        )

    cards = []
    for node in nodes:
        idx = walk_ids.index(node.id) + 1 if node.id in walk_ids else None
        cards.append(
            f"""
            <article class="node-card {'selected' if idx else ''}">
              <img src="{html.escape(rel(node.image_path))}" alt="{html.escape(node.title)}">
              <div>
                <span class="eyebrow">{'walk step ' + str(idx) if idx else node.cluster}</span>
                <h3>{html.escape(node.title)}</h3>
                <p>{html.escape(summarize(node.text, 180))}</p>
                <small>{html.escape(', '.join(node.keywords[:5]))}</small>
              </div>
            </article>
            """
        )

    walk_html = []
    for item in walk:
        walk_html.append(
            f"""
            <li>
              <span class="num">{item['step']}</span>
              <div>
                <b>💭 {html.escape(item['imagined'])}</b>
                <p>→ <strong>{html.escape(item['title'])}</strong> <em>score {item['score']:.2f}</em></p>
                <p>{html.escape(item['summary'])}</p>
              </div>
            </li>
            """
        )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Knowledge Image Wiki Prototype</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --ink: #17202a;
      --muted: #65727f;
      --line: #c9d0d5;
      --accent: #d34f2f;
      --blue: #246a8f;
      --panel: #fffdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      letter-spacing: 0;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid #dfd8cc;
      background: #fffaf1;
    }}
    h1 {{ margin: 0; font-size: 30px; line-height: 1.15; }}
    header p {{ max-width: 980px; color: var(--muted); font-size: 15px; line-height: 1.6; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1.1fr) 430px; min-height: calc(100vh - 134px); }}
    .graph-wrap {{ padding: 20px; overflow: auto; }}
    svg {{ width: 100%; min-width: 900px; height: 760px; background: #fdfaf4; border: 1px solid #ded7cc; }}
    .edge {{ stroke: var(--line); opacity: .78; }}
    .walk-edge {{ stroke: var(--accent); opacity: .98; }}
    .graph-node circle {{ fill: #fffdf8; stroke: #38546a; stroke-width: 3; }}
    .graph-node.walk-node circle {{ stroke: var(--accent); stroke-width: 5; fill: #fff4ed; }}
    .node-label {{ text-anchor: middle; font-size: 12px; fill: #24313d; font-weight: 700; paint-order: stroke; stroke: #fffaf1; stroke-width: 4px; }}
    .step-badge {{ font-size: 15px; font-weight: 800; fill: var(--accent); }}
    aside {{ border-left: 1px solid #dfd8cc; background: var(--panel); overflow: auto; max-height: calc(100vh - 134px); }}
    .section {{ padding: 22px; border-bottom: 1px solid #ebe5da; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .pipeline {{ display: grid; gap: 8px; }}
    .pipeline div {{ padding: 10px 12px; background: #f4efe5; border-left: 4px solid var(--blue); font-size: 14px; }}
    ol.walk {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 14px; }}
    ol.walk li {{ display: grid; grid-template-columns: 34px 1fr; gap: 10px; }}
    .num {{ width: 28px; height: 28px; border-radius: 50%; background: var(--accent); color: white; display: grid; place-items: center; font-weight: 800; }}
    .walk b {{ font-size: 13px; line-height: 1.35; }}
    .walk p {{ margin: 5px 0; font-size: 13px; line-height: 1.45; color: var(--muted); }}
    .cards {{ display: grid; gap: 10px; }}
    .node-card {{ display: grid; grid-template-columns: 86px 1fr; gap: 12px; padding: 10px; border: 1px solid #e4ddd2; background: #fff; }}
    .node-card.selected {{ border-color: var(--accent); box-shadow: inset 4px 0 0 var(--accent); }}
    .node-card img {{ width: 86px; height: 86px; object-fit: cover; }}
    .node-card h3 {{ margin: 2px 0 4px; font-size: 14px; }}
    .node-card p {{ margin: 0; color: var(--muted); font-size: 12px; line-height: 1.38; }}
    .node-card small, .eyebrow {{ display: block; color: var(--blue); font-size: 11px; font-weight: 700; margin-top: 5px; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Knowledge Upload → Keyword Images → AI Investigation Path</h1>
    <p>API 없이 만든 로컬 미니어처입니다. 문서를 넣고, 키워드를 뽑고, 키워드 이미지 카드를 만들고, 관련 노드를 그래프로 정렬한 뒤, 작업 질의에서 떠오른 시각적 단서가 필요한 지식 노드로 진입하는 흐름을 검증합니다.</p>
  </header>
  <main>
    <section class="graph-wrap">
      <svg viewBox="0 0 1200 760" role="img" aria-label="knowledge image graph">
        {''.join(edge_lines)}
        {''.join(node_svg)}
      </svg>
    </section>
    <aside>
      <section class="section">
        <h2>Pipeline</h2>
        <div class="pipeline">
          <div>1. 지식 문서 업로드: {len(nodes)} markdown chunks</div>
          <div>2. 키워드 자동 추출: TF-IDF top terms</div>
          <div>3. 키워드 이미지 카드 생성: deterministic visual handles</div>
          <div>4. 그래프 정렬: vector similarity + spring layout</div>
          <div>5. 작업 중 조사: imagined visual cue → nearest node → source text</div>
        </div>
      </section>
      <section class="section">
        <h2>AI Investigation Walk</h2>
        <ol class="walk">{''.join(walk_html)}</ol>
      </section>
      <section class="section">
        <h2>Knowledge Cards</h2>
        <div class="cards">{''.join(cards)}</div>
      </section>
    </aside>
  </main>
</body>
</html>
"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "knowledge_image_wiki.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def export_report(nodes: list[Node], graph: nx.Graph, walk: list[dict], html_path: Path, canvas_path: Path) -> Path:
    report = {
        "prototype": "knowledge-upload-to-keyword-image-wiki",
        "documents": len(nodes),
        "images": len(nodes),
        "edges": graph.number_of_edges(),
        "html": str(html_path.relative_to(ROOT)),
        "canvas": str(canvas_path.relative_to(ROOT)),
        "walk": walk,
        "note": "Local TF-IDF + deterministic image cards. Replace this vector layer with CLIP and card generation with GPT Image for production.",
    }
    path = OUT_DIR / "run_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    nodes = build_nodes()
    vectorizer, matrix, _ = vectorize(nodes)
    graph, _ = graph_from_vectors(nodes, matrix)
    walk = imagine_and_walk(nodes, matrix, graph)
    canvas_path = export_canvas(nodes, graph, walk)
    html_path = export_html(nodes, graph, walk)
    report_path = export_report(nodes, graph, walk, html_path, canvas_path)
    print(f"documents={len(nodes)}")
    print(f"keyword_images={len(nodes)}")
    print(f"edges={graph.number_of_edges()}")
    print(f"walk_steps={len(walk)}")
    print(f"html={html_path}")
    print(f"canvas={canvas_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
