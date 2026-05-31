from __future__ import annotations

import html
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import networkx as nx

from visual_memory_wiki.graphing import scale_positions
from visual_memory_wiki.models import Node, WalkStep, slug, summarize


def _rel_to(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def export_canvas(nodes: list[Node], graph: nx.Graph, walk: list[WalkStep], out_dir: Path) -> Path:
    vault_dir = Path(out_dir) / "my-wiki"
    image_dir = vault_dir / "keyword_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        if node.image_path:
            shutil.copy2(node.image_path, image_dir / node.image_path.name)
    positions = scale_positions(nodes, width=1800, height=1200)
    walk_ids = [step.node_id for step in walk]
    canvas = {"nodes": [], "edges": []}
    for node in nodes:
        x, y = positions[node.id]
        canvas["nodes"].append(
            {
                "id": node.id,
                "type": "file",
                "file": str(Path("keyword_images") / Path(node.image_path or "").name),
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
        canvas["edges"].append(
            {
                "id": f"e_{a}_{b}",
                "fromNode": a,
                "toNode": b,
                "color": "4" if a in walk_ids and b in walk_ids else "1",
                "label": f"{data.get('weight', 0):.2f}",
            }
        )
    path = vault_dir / "visual-memory-wiki.canvas"
    path.write_text(json.dumps(canvas, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_obsidian_notes(nodes: list[Node], walk: list[WalkStep], out_dir: Path, query: str = "") -> Path:
    vault_dir = Path(out_dir) / "my-wiki"
    notes_dir = vault_dir / "notes"
    sources_dir = vault_dir / "sources"
    notes_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    note_by_node: dict[str, str] = {}
    for node in nodes:
        source_copy = sources_dir / node.source_path.name
        shutil.copy2(node.source_path, source_copy)
        note_name = f"{node.id}-{slug(node.title)}.md"
        note_by_node[node.id] = note_name
        image_line = f"![[keyword_images/{Path(node.image_path or '').name}]]" if node.image_path else ""
        body = f"""# {node.title}

{image_line}

## Keywords
{", ".join(node.keywords)}

## Source
[[sources/{source_copy.name}]]

## Summary
{summarize(node.text, 520)}

## Role
This image is a navigation handle. The source text remains the ground truth.
"""
        (notes_dir / note_name).write_text(body, encoding="utf-8")
    lines = ["# Visual Retrieval Walk", "", f"Query: {query}", ""]
    for step in walk:
        note_name = note_by_node.get(step.node_id, "")
        lines.extend(
            [
                f"## {step.step}. [[notes/{note_name}|{step.title}]]",
                "",
                f"- Visual cue: {step.cue}",
                f"- Score: {step.score:.2f}",
                f"- Keywords: {', '.join(step.keywords[:5])}",
                f"- Summary: {step.summary}",
                "",
            ]
        )
    path = vault_dir / "visual-retrieval-walk.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_html(nodes: list[Node], graph: nx.Graph, walk: list[WalkStep], out_dir: Path, title: str = "Visual Memory Wiki") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions = scale_positions(nodes)
    walk_ids = [step.node_id for step in walk]
    edge_lines = []
    for a, b, data in graph.edges(data=True):
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        is_walk = a in walk_ids and b in walk_ids
        edge_lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge {"walk-edge" if is_walk else ""}" stroke-width="{2 + data.get("weight", 0) * 8:.1f}" />'
        )
    node_svg = []
    for node in nodes:
        x, y = positions[node.id]
        idx = walk_ids.index(node.id) + 1 if node.id in walk_ids else None
        image_href = _rel_to(node.image_path, out) if node.image_path else ""
        node_svg.append(
            f"""
            <g class="graph-node {'walk-node' if idx else ''}" transform="translate({x:.1f},{y:.1f})">
              <circle r="{34 if idx else 26}"></circle>
              <image href="{html.escape(image_href)}" x="-24" y="-24" width="48" height="48"></image>
              {'<text class="step-badge" x="27" y="-24">' + str(idx) + '</text>' if idx else ''}
              <text class="node-label" y="51">{html.escape(node.title)}</text>
            </g>
            """
        )
    walk_items = []
    for step in walk:
        walk_items.append(
            f"""
            <li>
              <span class="num">{step.step}</span>
              <div>
                <b>{html.escape(step.cue)}</b>
                <p>→ <strong>{html.escape(step.title)}</strong> <em>score {step.score:.2f}</em></p>
                <p>{html.escape(step.summary)}</p>
              </div>
            </li>
            """
        )
    cards = []
    for node in nodes:
        image_src = _rel_to(node.image_path, out) if node.image_path else ""
        cards.append(
            f"""
            <article class="node-card">
              <img src="{html.escape(image_src)}" alt="{html.escape(node.title)}">
              <div>
                <h3>{html.escape(node.title)}</h3>
                <p>{html.escape(summarize(node.text, 170))}</p>
                <small>{html.escape(', '.join(node.keywords[:5]))}</small>
              </div>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --bg:#f7f4ee; --ink:#17202a; --muted:#65727f; --line:#c9d0d5; --accent:#d34f2f; --blue:#246a8f; --panel:#fffdf8; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--bg); letter-spacing:0; }}
    header {{ padding:28px 32px 18px; border-bottom:1px solid #dfd8cc; background:#fffaf1; }}
    h1 {{ margin:0; font-size:30px; line-height:1.15; }}
    header p {{ max-width:980px; color:var(--muted); font-size:15px; line-height:1.6; }}
    main {{ display:grid; grid-template-columns:minmax(0, 1.1fr) 430px; min-height:calc(100vh - 134px); }}
    .graph-wrap {{ padding:20px; overflow:auto; }}
    svg {{ width:100%; min-width:900px; height:760px; background:#fdfaf4; border:1px solid #ded7cc; }}
    .edge {{ stroke:var(--line); opacity:.78; }}
    .walk-edge {{ stroke:var(--accent); opacity:.98; }}
    .graph-node circle {{ fill:#fffdf8; stroke:#38546a; stroke-width:3; }}
    .graph-node.walk-node circle {{ stroke:var(--accent); stroke-width:5; fill:#fff4ed; }}
    .node-label {{ text-anchor:middle; font-size:12px; fill:#24313d; font-weight:700; paint-order:stroke; stroke:#fffaf1; stroke-width:4px; }}
    .step-badge {{ font-size:15px; font-weight:800; fill:var(--accent); }}
    aside {{ border-left:1px solid #dfd8cc; background:var(--panel); overflow:auto; max-height:calc(100vh - 134px); }}
    .section {{ padding:22px; border-bottom:1px solid #ebe5da; }}
    h2 {{ margin:0 0 14px; font-size:18px; }}
    ol.walk {{ list-style:none; margin:0; padding:0; display:grid; gap:14px; }}
    ol.walk li {{ display:grid; grid-template-columns:34px 1fr; gap:10px; }}
    .num {{ width:28px; height:28px; border-radius:50%; background:var(--accent); color:white; display:grid; place-items:center; font-weight:800; }}
    .walk b {{ font-size:13px; line-height:1.35; }}
    .walk p {{ margin:5px 0; font-size:13px; line-height:1.45; color:var(--muted); }}
    .cards {{ display:grid; gap:10px; }}
    .node-card {{ display:grid; grid-template-columns:86px 1fr; gap:12px; padding:10px; border:1px solid #e4ddd2; background:#fff; }}
    .node-card img {{ width:86px; height:86px; object-fit:cover; }}
    .node-card h3 {{ margin:2px 0 4px; font-size:14px; }}
    .node-card p {{ margin:0; color:var(--muted); font-size:12px; line-height:1.38; }}
    .node-card small {{ display:block; color:var(--blue); font-size:11px; font-weight:700; margin-top:5px; }}
    @media (max-width:980px) {{ main {{ grid-template-columns:1fr; }} aside {{ border-left:0; max-height:none; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Markdown notes become visual cards, CLIP or TF-IDF places them into a graph, and short visual cues retrieve source-grounded knowledge nodes.</p>
  </header>
  <main>
    <section class="graph-wrap"><svg viewBox="0 0 1200 760">{''.join(edge_lines)}{''.join(node_svg)}</svg></section>
    <aside>
      <section class="section"><h2>Visual Retrieval Walk</h2><ol class="walk">{''.join(walk_items)}</ol></section>
      <section class="section"><h2>Knowledge Cards</h2><div class="cards">{''.join(cards)}</div></section>
    </aside>
  </main>
</body>
</html>
"""
    path = out / "visual-memory-wiki.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def write_json_report(path: Path, data: dict) -> Path:
    def default(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj))

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")
    return path
