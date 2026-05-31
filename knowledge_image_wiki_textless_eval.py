#!/usr/bin/env python3
"""
Evaluate whether the image-wiki retrieval still works after removing visible
text from the keyword cards.

It compares three routes over the same six imagined visual cues:

1. text_baseline: TF-IDF over the source document text.
2. text_card_clip: CLIP text cue -> CLIP image embedding of cards that contain text.
3. textless_card_clip: CLIP text cue -> CLIP image embedding of text-free cards.

The point is not to maximize scores. The point is to expose whether the previous
CLIP result was mostly OCR-like text matching or genuinely visual.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity

import knowledge_image_wiki_clip_pipeline as clip_pipe
import knowledge_image_wiki_demo as demo


EXPECTED = {
    1: "n02",
    2: "n01",
    3: "n03",
    4: "n04",
    5: "n07",
    6: "n08",
}


MODE_LABELS = {
    "text_baseline": "Text search over source notes",
    "text_card_clip": "CLIP over cards with visible text",
    "textless_card_clip": "CLIP over text-free visual cards",
    "generated_card_clip": "CLIP over generated text-free image cards",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare text vs text-card vs textless-card retrieval.")
    parser.add_argument("--docs", type=Path, default=demo.ROOT / "demo_knowledge")
    parser.add_argument("--out", type=Path, default=demo.ROOT / "generated_textless_eval")
    parser.add_argument("--model", default="sentence-transformers/clip-ViT-B-32")
    parser.add_argument("--generated-card-dir", type=Path)
    return parser.parse_args()


def configure(docs: Path, out: Path) -> None:
    demo.DOC_DIR = docs.resolve()
    demo.OUT_DIR = out.resolve()
    demo.IMG_DIR = demo.OUT_DIR / "text_cards"
    demo.VAULT_DIR = demo.OUT_DIR / "my-wiki"


def palette(node_id: str) -> tuple[str, str, str, str]:
    palettes = [
        ("#102A43", "#F0F4F8", "#F59E0B", "#38BDF8"),
        ("#19381F", "#F7F3E8", "#D95D39", "#7FB069"),
        ("#2D3142", "#F5F1ED", "#4F9DDE", "#EF8354"),
        ("#16324F", "#F8F7F2", "#D1495B", "#56A3A6"),
        ("#2F2D2E", "#FAF3DD", "#227C9D", "#FE6D73"),
        ("#263238", "#ECEFF1", "#8BC34A", "#FFB300"),
        ("#352F44", "#F9F7F7", "#DB504A", "#4ECDC4"),
        ("#233D4D", "#FEF9EF", "#FCCA46", "#619B8A"),
    ]
    idx = int("".join(ch for ch in node_id if ch.isdigit()) or "1") - 1
    return palettes[idx % len(palettes)]


def draw_card_frame(draw: ImageDraw.ImageDraw, bg: str, paper: str, accent: str) -> None:
    draw.rounded_rectangle((22, 22, 490, 490), radius=28, fill=paper, outline=accent, width=5)
    draw.rounded_rectangle((42, 42, 470, 470), radius=18, outline=bg, width=3)


def draw_doorway(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FAF7F0", accent)
    draw.rounded_rectangle((178, 92, 334, 352), radius=18, outline=ink, width=10)
    draw.rectangle((204, 126, 308, 352), fill=bg)
    for r in range(34, 166, 30):
        draw.arc((256 - r, 222 - r, 256 + r, 222 + r), 35, 325, fill=alt, width=5)
    points = [(98, 392), (172, 354), (256, 382), (340, 350), (418, 392)]
    draw.line(points, fill=ink, width=9, joint="curve")
    for x, y in points:
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=accent)


def draw_upload_cards(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FBF8EF", accent)
    for offset in [0, 18, 36]:
        draw.rounded_rectangle((92 + offset, 92 + offset, 246 + offset, 246 + offset), radius=12, fill="#FFFFFF", outline=ink, width=5)
    draw.line((168, 312, 168, 382), fill=accent, width=13)
    draw.polygon([(168, 266), (132, 324), (204, 324)], fill=accent)
    for i, (x, y) in enumerate([(300, 116), (360, 178), (302, 252), (384, 310)]):
        color = alt if i % 2 else "#FFFFFF"
        draw.rounded_rectangle((x, y, x + 84, y + 58), radius=10, fill=color, outline=ink, width=4)
    draw.line((248, 198, 300, 146), fill=ink, width=5)
    draw.line((250, 220, 360, 202), fill=ink, width=5)
    draw.line((240, 242, 302, 280), fill=ink, width=5)


def draw_embedding_map(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#F7FAFC", accent)
    for x in range(96, 430, 56):
        draw.line((x, 96, x, 414), fill="#D7DEE8", width=2)
    for y in range(96, 430, 56):
        draw.line((96, y, 416, y), fill="#D7DEE8", width=2)
    draw.line((104, 404, 420, 404), fill=ink, width=6)
    draw.line((112, 416, 112, 92), fill=ink, width=6)
    clusters = [
        [(172, 162), (198, 188), (150, 210)],
        [(310, 152), (346, 186), (290, 214)],
        [(250, 298), (286, 326), (220, 346)],
    ]
    colors = [accent, alt, "#7FB069"]
    for group, color in zip(clusters, colors):
        for x, y in group:
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=color, outline=ink, width=3)
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            draw.line((group[a][0], group[a][1], group[b][0], group[b][1]), fill=color, width=4)


def draw_walker(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FDF8F0", accent)
    draw.arc((78, 296, 436, 490), 190, 348, fill=ink, width=9)
    draw.ellipse((208, 124, 258, 174), fill=ink)
    draw.line((232, 176, 232, 270), fill=ink, width=12)
    draw.line((232, 212, 186, 252), fill=ink, width=10)
    draw.line((232, 212, 284, 244), fill=ink, width=10)
    draw.line((232, 270, 194, 348), fill=ink, width=12)
    draw.line((232, 270, 286, 342), fill=ink, width=12)
    draw.line((284, 244, 334, 210), fill=ink, width=6)
    draw.ellipse((322, 190, 372, 240), fill=accent, outline=ink, width=5)
    for r in [48, 76, 104]:
        draw.arc((347 - r, 215 - r, 347 + r, 215 + r), 300, 60, fill=alt, width=4)
    for x, y in [(104, 338), (138, 318), (392, 326), (420, 306)]:
        draw.rounded_rectangle((x, y, x + 42, y + 30), radius=6, fill="#FFFFFF", outline=ink, width=3)


def draw_canvas_board(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FFFBF0", accent)
    draw.rectangle((84, 92, 428, 382), fill="#F8F2E4", outline=ink, width=6)
    cards = [(122, 126), (278, 136), (178, 242), (316, 278)]
    centers = []
    for i, (x, y) in enumerate(cards):
        draw.rounded_rectangle((x, y, x + 92, y + 64), radius=10, fill="#FFFFFF" if i % 2 else alt, outline=ink, width=4)
        centers.append((x + 46, y + 32))
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        draw.line((centers[a][0], centers[a][1], centers[b][0], centers[b][1]), fill=accent, width=5)
    draw.ellipse((388, 390, 426, 428), fill=accent, outline=ink, width=4)
    draw.line((406, 392, 406, 342), fill=ink, width=5)


def draw_hardware(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#F8FAF6", accent)
    draw.rounded_rectangle((94, 104, 418, 396), radius=18, fill="#254441", outline=ink, width=7)
    for x in range(126, 398, 34):
        draw.line((x, 132, x, 368), fill=alt, width=5)
    for y in range(136, 368, 38):
        draw.line((122, y, 390, y), fill="#6EA77A", width=4)
    for x, y in [(164, 168), (242, 182), (324, 158), (180, 290), (286, 300), (354, 266)]:
        draw.ellipse((x - 15, y - 15, x + 15, y + 15), fill=accent, outline="#F8FAF6", width=3)
    for x in range(112, 404, 24):
        draw.line((x, 410, x + 8, 448), fill=ink, width=4)


def draw_evidence(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FFF8EA", accent)
    draw.rectangle((86, 94, 426, 366), fill="#E8D8B5", outline=ink, width=7)
    pins = [(148, 142), (302, 130), (198, 250), (352, 264)]
    for i, (x, y) in enumerate(pins):
        draw.rounded_rectangle((x - 36, y - 24, x + 42, y + 38), radius=6, fill="#FFFFFF", outline=ink, width=3)
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=accent)
    for a, b in [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3)]:
        draw.line((pins[a][0], pins[a][1], pins[b][0], pins[b][1]), fill=accent, width=4)
    draw.ellipse((294, 326, 374, 406), outline=alt, width=9)
    draw.line((356, 388, 414, 446), fill=alt, width=11)


def draw_safety(draw: ImageDraw.ImageDraw, bg: str, ink: str, accent: str, alt: str) -> None:
    draw_card_frame(draw, bg, "#FFFBF4", accent)
    draw.polygon([(256, 88), (394, 150), (374, 332), (256, 410), (138, 332), (118, 150)], fill="#E7F6EF", outline=ink, width=8)
    draw.polygon([(256, 146), (350, 320), (162, 320)], fill=accent, outline=ink, width=7)
    draw.line((256, 202, 256, 274), fill="#FFFBF4", width=13)
    draw.ellipse((248, 292, 264, 308), fill="#FFFBF4")
    draw.ellipse((206, 352, 306, 424), outline=alt, width=8)
    draw.line((256, 352, 256, 292), fill=alt, width=8)


DRAWERS = {
    "n01": draw_doorway,
    "n02": draw_upload_cards,
    "n03": draw_embedding_map,
    "n04": draw_walker,
    "n05": draw_canvas_board,
    "n06": draw_hardware,
    "n07": draw_evidence,
    "n08": draw_safety,
}


def generate_textless_card(node: demo.Node, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    bg, paper, accent, alt = palette(node.id)
    image = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(image)
    drawer = DRAWERS.get(node.id, draw_doorway)
    drawer(draw, bg, "#17202A", accent, alt)
    path = out_dir / f"{node.id}_{demo.slug(node.title)}_textless.png"
    image.save(path)
    return path


def copy_nodes_with_textless_images(nodes: list[demo.Node], image_dir: Path) -> list[demo.Node]:
    textless: list[demo.Node] = []
    for node in nodes:
        path = generate_textless_card(node, image_dir)
        textless.append(replace(node, image_path=path))
    return textless


def copy_nodes_with_external_images(nodes: list[demo.Node], image_dir: Path, out_dir: Path) -> list[demo.Node]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[demo.Node] = []
    for node in nodes:
        matches = sorted(image_dir.glob(f"{node.id}_*.png")) + sorted(image_dir.glob(f"{node.id}_*.jpg")) + sorted(image_dir.glob(f"{node.id}_*.jpeg"))
        if not matches:
            raise FileNotFoundError(f"No generated image found for {node.id} in {image_dir}")
        src = matches[0]
        dst = out_dir / f"{node.id}_{demo.slug(node.title)}_generated{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(replace(node, image_path=dst))
    return copied


def load_clip_model(model_name: str) -> Any:
    return clip_pipe.load_clip_model(model_name)


def encode_images(model: Any, nodes: list[demo.Node]) -> np.ndarray:
    return clip_pipe.encode_clip_images(model, nodes)


def encode_text(model: Any, phrases: list[str]) -> np.ndarray:
    return clip_pipe.encode_clip_text(model, phrases)


def text_baseline(nodes: list[demo.Node], cues: list[str]) -> dict:
    _, node_matrix, cue_matrix = demo.vectorize(nodes, cues)
    scores = cosine_similarity(cue_matrix, node_matrix)
    return score_table(nodes, cues, scores)


def clip_image_mode(model: Any, nodes: list[demo.Node], cues: list[str]) -> dict:
    node_matrix = encode_images(model, nodes)
    cue_matrix = encode_text(model, cues)
    scores = cosine_similarity(cue_matrix, node_matrix)
    return score_table(nodes, cues, scores)


def score_table(nodes: list[demo.Node], cues: list[str], scores: np.ndarray) -> dict:
    rows = []
    reciprocal_ranks = []
    hits = 0
    node_ids = [node.id for node in nodes]
    for idx, cue in enumerate(cues, start=1):
        expected = EXPECTED[idx]
        ranked = np.argsort(scores[idx - 1])[::-1]
        rank_ids = [node_ids[int(i)] for i in ranked]
        top_idx = int(ranked[0])
        top_node = nodes[top_idx]
        expected_rank = rank_ids.index(expected) + 1
        reciprocal_ranks.append(1 / expected_rank)
        correct = top_node.id == expected
        hits += int(correct)
        rows.append(
            {
                "step": idx,
                "cue": cue,
                "expected_id": expected,
                "expected_title": nodes[node_ids.index(expected)].title,
                "top_id": top_node.id,
                "top_title": top_node.title,
                "top_score": float(scores[idx - 1, top_idx]),
                "expected_rank": expected_rank,
                "correct": correct,
                "top3": [
                    {
                        "id": nodes[int(i)].id,
                        "title": nodes[int(i)].title,
                        "score": float(scores[idx - 1, int(i)]),
                    }
                    for i in ranked[:3]
                ],
            }
        )
    return {
        "accuracy": hits / len(cues),
        "mrr": float(np.mean(reciprocal_ranks)),
        "rows": rows,
    }


def graph_for_nodes(model: Any, nodes: list[demo.Node]) -> nx.Graph:
    matrix = encode_images(model, nodes)
    graph, _ = demo.graph_from_vectors(nodes, matrix)
    return graph


def export_eval_html(out: Path, results: dict, text_nodes: list[demo.Node], textless_nodes: list[demo.Node]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    image_grid = []
    for text_node, textless_node in zip(text_nodes, textless_nodes):
        image_grid.append(
            f"""
            <article>
              <h3>{html.escape(text_node.title)}</h3>
              <div class="pair">
                <figure><img src="{html.escape(str(text_node.image_path.relative_to(out)))}"><figcaption>with text</figcaption></figure>
                <figure><img src="{html.escape(str(textless_node.image_path.relative_to(out)))}"><figcaption>text-free</figcaption></figure>
              </div>
            </article>
            """
        )

    sections = []
    for mode, data in results.items():
        rows = []
        for row in data["rows"]:
            top3 = "<br>".join(
                f"{html.escape(item['title'])} <span>{item['score']:.2f}</span>" for item in row["top3"]
            )
            rows.append(
                f"""
                <tr class="{'ok' if row['correct'] else 'miss'}">
                  <td>{row['step']}</td>
                  <td>{html.escape(row['cue'])}</td>
                  <td>{html.escape(row['expected_title'])}</td>
                  <td>{html.escape(row['top_title'])}<br><small>rank expected: {row['expected_rank']} · score {row['top_score']:.2f}</small></td>
                  <td>{top3}</td>
                </tr>
                """
            )
        sections.append(
            f"""
            <section>
              <h2>{html.escape(MODE_LABELS[mode])}</h2>
              <p class="metric">Accuracy {data['accuracy']:.0%} · MRR {data['mrr']:.2f}</p>
              <table>
                <thead><tr><th>Step</th><th>Imagined cue</th><th>Expected</th><th>Top result</th><th>Top 3</th></tr></thead>
                <tbody>{''.join(rows)}</tbody>
              </table>
            </section>
            """
        )

    verdict = verdict_text(results)
    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Textless Card Evaluation</title>
  <style>
    :root {{ --bg:#f8f5ee; --ink:#17202a; --muted:#66717d; --line:#ded6ca; --ok:#e7f4ea; --miss:#fff0ed; --accent:#cf4f31; --blue:#246a8f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }}
    header {{ padding:30px 34px 18px; background:#fffaf1; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:30px; }}
    header p {{ max-width:1040px; line-height:1.55; color:var(--muted); }}
    main {{ padding:26px 30px 48px; display:grid; gap:26px; }}
    .verdict {{ background:#fff; border-left:5px solid var(--accent); padding:18px 20px; max-width:1040px; line-height:1.55; }}
    section {{ background:#fffdf8; border:1px solid var(--line); padding:18px; overflow:auto; }}
    h2 {{ margin:0 0 8px; font-size:20px; }}
    .metric {{ color:var(--blue); font-weight:800; margin:0 0 12px; }}
    table {{ width:100%; min-width:920px; border-collapse:collapse; font-size:13px; }}
    th,td {{ text-align:left; vertical-align:top; padding:10px 12px; border-bottom:1px solid #e9e1d7; line-height:1.4; }}
    th {{ background:#f0e9dd; }}
    tr.ok {{ background:var(--ok); }}
    tr.miss {{ background:var(--miss); }}
    small, span {{ color:var(--muted); }}
    .images {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }}
    article {{ background:#fff; border:1px solid var(--line); padding:12px; }}
    article h3 {{ margin:0 0 10px; font-size:15px; }}
    .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    figure {{ margin:0; }}
    img {{ width:100%; aspect-ratio:1; object-fit:cover; border:1px solid #ddd2c4; }}
    figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Textless Keyword Image Evaluation</h1>
    <p>이 검증은 이전 결과가 카드 안의 글자에 기대고 있었는지 확인합니다. 같은 지식 노드와 같은 상상 단서를 두고, 소스 텍스트 검색, 글자 포함 카드의 CLIP 검색, 글자 없는 순수 이미지 카드의 CLIP 검색을 비교합니다.</p>
  </header>
  <main>
    <div class="verdict">{html.escape(verdict)}</div>
    {''.join(sections)}
    <section>
      <h2>Card Pairs</h2>
      <div class="images">{''.join(image_grid)}</div>
    </section>
  </main>
</body>
</html>
"""
    path = out / "textless_eval.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def verdict_text(results: dict) -> str:
    text_card = results["text_card_clip"]["accuracy"]
    textless = results["textless_card_clip"]["accuracy"]
    baseline = results["text_baseline"]["accuracy"]
    if textless >= text_card:
        return (
            "글자 없는 카드가 글자 포함 카드와 같거나 더 잘 맞았습니다. 이 경우 시각 표상이 실제로 검색 신호가 된다는 강한 증거입니다."
        )
    if textless >= 0.5:
        return (
            "글자 없는 카드도 절반 이상 맞췄지만 글자 포함 카드보다 약합니다. 구조는 가능하지만, 이미지 생성 품질이 검색 품질을 좌우합니다."
        )
    if textless < 0.5 and (text_card > textless or baseline > textless):
        return (
            "글자 없는 카드는 약합니다. 이전 성공의 상당 부분은 카드 안의 텍스트 또는 텍스트와 결합된 시각 단서였을 가능성이 큽니다."
        )
    if text_card > textless and text_card >= baseline:
        return (
            "글자 포함 카드가 강하고 글자 없는 카드는 약합니다. 이전 성공의 상당 부분은 카드 안의 텍스트 또는 텍스트와 결합된 시각 단서였을 가능성이 큽니다."
        )
    return (
        "세 모드 모두 뚜렷하게 우세하지 않습니다. 현재 샘플이나 상상 단서가 작아서 더 큰 실제 노트로 재검증해야 합니다."
    )


def verdict_text_multi(scenario_results: dict) -> str:
    strict = scenario_results["cue_only"]
    contextual = scenario_results["task_context"]
    strict_textless = strict["textless_card_clip"]["accuracy"]
    context_textless = contextual["textless_card_clip"]["accuracy"]
    context_text_card = contextual["text_card_clip"]["accuracy"]
    if "generated_card_clip" in strict:
        strict_generated = strict["generated_card_clip"]["accuracy"]
        context_generated = contextual["generated_card_clip"]["accuracy"]
        if strict_generated >= 0.66 and context_generated < strict_generated:
            return (
                "생성 이미지 카드는 순수 시각 단서에서 강하게 개선됐지만, 긴 작업 질문을 그대로 붙이면 성능이 떨어집니다. "
                "따라서 제품형 구조는 '작업 맥락을 LLM이 짧은 시각 단서로 증류 → CLIP 검색'이 맞고, 원시 작업 문장을 CLIP에 직접 섞으면 잡음이 됩니다."
            )
        if strict_generated > strict_textless:
            return (
                "생성 이미지 카드가 손그림 텍스트 없는 카드보다 낫습니다. 이미지 생성 품질이 병목이라는 가설이 지지됩니다."
            )
    if strict_textless >= 0.5 and context_textless >= 0.5:
        return "텍스트 없는 카드가 순수 단서와 작업 맥락 모두에서 절반 이상 맞췄습니다. 순수 이미지 표상은 약하지만 실제 신호로 작동합니다."
    if strict_textless < 0.5 <= context_textless:
        return "순수 시각 단서만으로는 약하지만, 작업 질문 맥락이 붙으면 텍스트 없는 카드도 절반까지 올라갑니다. 제품형 구조는 가능하나 이미지 생성 품질과 작업 맥락이 필수입니다."
    if context_text_card > context_textless:
        return "글자 포함 카드가 텍스트 없는 카드보다 뚜렷하게 강합니다. 현재 단계에서는 이미지 표상만으로 충분하다고 보기 어렵고, 이전 성공에는 카드 텍스트 의존이 섞여 있습니다."
    return "결과가 애매합니다. 더 큰 실제 노트와 더 풍부한 생성 이미지로 재검증해야 합니다."


def export_eval_html_multi(
    out: Path,
    scenario_results: dict,
    text_nodes: list[demo.Node],
    textless_nodes: list[demo.Node],
    generated_nodes: list[demo.Node] | None = None,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    image_grid = []
    for text_node, textless_node in zip(text_nodes, textless_nodes):
        generated_html = ""
        if generated_nodes is not None:
            generated_node = next(node for node in generated_nodes if node.id == text_node.id)
            generated_html = f"""
                <figure><img src="{html.escape(str(generated_node.image_path.relative_to(out)))}"><figcaption>generated</figcaption></figure>
            """
        image_grid.append(
            f"""
            <article>
              <h3>{html.escape(text_node.title)}</h3>
              <div class="pair {'triple' if generated_nodes is not None else ''}">
                <figure><img src="{html.escape(str(text_node.image_path.relative_to(out)))}"><figcaption>with text</figcaption></figure>
                <figure><img src="{html.escape(str(textless_node.image_path.relative_to(out)))}"><figcaption>text-free</figcaption></figure>
                {generated_html}
              </div>
            </article>
            """
        )

    scenario_sections = []
    for scenario, results in scenario_results.items():
        sections = []
        for mode, data in results.items():
            rows = []
            for row in data["rows"]:
                top3 = "<br>".join(
                    f"{html.escape(item['title'])} <span>{item['score']:.2f}</span>" for item in row["top3"]
                )
                rows.append(
                    f"""
                    <tr class="{'ok' if row['correct'] else 'miss'}">
                      <td>{row['step']}</td>
                      <td>{html.escape(row['cue'])}</td>
                      <td>{html.escape(row['expected_title'])}</td>
                      <td>{html.escape(row['top_title'])}<br><small>rank expected: {row['expected_rank']} · score {row['top_score']:.2f}</small></td>
                      <td>{top3}</td>
                    </tr>
                    """
                )
            sections.append(
                f"""
                <section>
                  <h3>{html.escape(MODE_LABELS[mode])}</h3>
                  <p class="metric">Accuracy {data['accuracy']:.0%} · MRR {data['mrr']:.2f}</p>
                  <table>
                    <thead><tr><th>Step</th><th>Imagined cue</th><th>Expected</th><th>Top result</th><th>Top 3</th></tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                  </table>
                </section>
                """
            )
        title = "Cue only" if scenario == "cue_only" else "Task context + cue"
        note = (
            "상상한 시각 단서만 던진 엄격한 조건입니다."
            if scenario == "cue_only"
            else "실제 작업 중처럼 전체 질문과 상상 단서를 함께 던진 조건입니다."
        )
        scenario_sections.append(
            f"""
            <div class="scenario">
              <h2>{title}</h2>
              <p>{note}</p>
              {''.join(sections)}
            </div>
            """
        )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Textless Card Evaluation</title>
  <style>
    :root {{ --bg:#f8f5ee; --ink:#17202a; --muted:#66717d; --line:#ded6ca; --ok:#e7f4ea; --miss:#fff0ed; --accent:#cf4f31; --blue:#246a8f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; letter-spacing:0; }}
    header {{ padding:30px 34px 18px; background:#fffaf1; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 10px; font-size:30px; }}
    header p {{ max-width:1040px; line-height:1.55; color:var(--muted); }}
    main {{ padding:26px 30px 48px; display:grid; gap:28px; }}
    .verdict {{ background:#fff; border-left:5px solid var(--accent); padding:18px 20px; max-width:1080px; line-height:1.55; }}
    .scenario {{ display:grid; gap:16px; }}
    .scenario > h2 {{ margin:0; font-size:24px; }}
    .scenario > p {{ margin:0; color:var(--muted); }}
    section {{ background:#fffdf8; border:1px solid var(--line); padding:18px; overflow:auto; }}
    h3 {{ margin:0 0 8px; font-size:18px; }}
    .metric {{ color:var(--blue); font-weight:800; margin:0 0 12px; }}
    table {{ width:100%; min-width:920px; border-collapse:collapse; font-size:13px; }}
    th,td {{ text-align:left; vertical-align:top; padding:10px 12px; border-bottom:1px solid #e9e1d7; line-height:1.4; }}
    th {{ background:#f0e9dd; }}
    tr.ok {{ background:var(--ok); }}
    tr.miss {{ background:var(--miss); }}
    small, span {{ color:var(--muted); }}
    .images {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }}
    article {{ background:#fff; border:1px solid var(--line); padding:12px; }}
    article h3 {{ margin:0 0 10px; font-size:15px; }}
    .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .pair.triple {{ grid-template-columns:1fr 1fr 1fr; }}
    figure {{ margin:0; }}
    img {{ width:100%; aspect-ratio:1; object-fit:cover; border:1px solid #ddd2c4; }}
    figcaption {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Textless Keyword Image Evaluation</h1>
    <p>이 검증은 이전 결과가 카드 안의 글자에 기대고 있었는지 확인합니다. 같은 지식 노드와 같은 상상 단서를 두고, 소스 텍스트 검색, 글자 포함 카드의 CLIP 검색, 글자 없는 순수 이미지 카드의 CLIP 검색을 비교합니다.</p>
  </header>
  <main>
    <div class="verdict">{html.escape(verdict_text_multi(scenario_results))}</div>
    {''.join(scenario_sections)}
    <section>
      <h3>Card Pairs</h3>
      <div class="images">{''.join(image_grid)}</div>
    </section>
  </main>
</body>
</html>
"""
    path = out / "textless_eval.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def export_report(
    out: Path,
    results: dict,
    html_path: Path,
    text_nodes: list[demo.Node],
    textless_nodes: list[demo.Node],
    generated_nodes: list[demo.Node] | None = None,
) -> Path:
    report = {
        "prototype": "textless-keyword-image-evaluation",
        "expected": EXPECTED,
        "modes": MODE_LABELS,
        "results": results,
        "html": str(html_path),
        "text_card_images": [str(node.image_path) for node in text_nodes],
        "textless_card_images": [str(node.image_path) for node in textless_nodes],
        "generated_card_images": [str(node.image_path) for node in generated_nodes] if generated_nodes else [],
        "verdict": verdict_text_multi(results) if "cue_only" in results else verdict_text(results),
    }
    path = out / "textless_eval_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_best_mode_wiki(out: Path, model: Any, textless_nodes: list[demo.Node], results: dict) -> tuple[Path, Path]:
    graph = graph_for_nodes(model, textless_nodes)
    walk = []
    by_id = {node.id: node for node in textless_nodes}
    for row in results["textless_card_clip"]["rows"]:
        node = by_id[row["top_id"]]
        walk.append(
            {
                "step": row["step"],
                "imagined": row["cue"],
                "node_id": node.id,
                "title": node.title,
                "score": row["top_score"],
                "keywords": node.keywords,
                "source": str(node.path),
                "summary": demo.summarize(node.text),
                "neighbors": [],
            }
        )
    canvas_path = demo.export_canvas(textless_nodes, graph, walk)
    html_path = demo.export_html(textless_nodes, graph, walk)
    return html_path, canvas_path


def main() -> None:
    args = parse_args()
    configure(args.docs, args.out)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    text_nodes = demo.build_nodes()
    textless_nodes = copy_nodes_with_textless_images(text_nodes, out / "textless_cards")
    generated_nodes = None
    if args.generated_card_dir:
        generated_nodes = copy_nodes_with_external_images(
            text_nodes,
            args.generated_card_dir.resolve(),
            out / "generated_cards",
        )
    model = load_clip_model(args.model)
    cue_only = clip_pipe.IMAGINED_STEPS
    task_context = [f"{clip_pipe.DEFAULT_QUERY}. Visual target: {cue}" for cue in clip_pipe.IMAGINED_STEPS]

    results = {
        "cue_only": {
            "text_baseline": text_baseline(text_nodes, cue_only),
            "text_card_clip": clip_image_mode(model, text_nodes, cue_only),
            "textless_card_clip": clip_image_mode(model, textless_nodes, cue_only),
        },
        "task_context": {
            "text_baseline": text_baseline(text_nodes, task_context),
            "text_card_clip": clip_image_mode(model, text_nodes, task_context),
            "textless_card_clip": clip_image_mode(model, textless_nodes, task_context),
        },
    }
    if generated_nodes is not None:
        results["cue_only"]["generated_card_clip"] = clip_image_mode(model, generated_nodes, cue_only)
        results["task_context"]["generated_card_clip"] = clip_image_mode(model, generated_nodes, task_context)

    html_path = export_eval_html_multi(out, results, text_nodes, textless_nodes, generated_nodes)
    report_path = export_report(out, results, html_path, text_nodes, textless_nodes, generated_nodes)
    wiki_nodes = generated_nodes if generated_nodes is not None else textless_nodes
    wiki_scenario = "task_context"
    if generated_nodes is not None:
        cue_acc = results["cue_only"]["generated_card_clip"]["accuracy"]
        context_acc = results["task_context"]["generated_card_clip"]["accuracy"]
        if cue_acc >= context_acc:
            wiki_scenario = "cue_only"
    textless_html, textless_canvas = export_best_mode_wiki(out, model, wiki_nodes, results[wiki_scenario])

    print(f"html={html_path}")
    print(f"report={report_path}")
    print(f"textless_wiki_html={textless_html}")
    print(f"textless_canvas={textless_canvas}")
    print(f"wiki_scenario={wiki_scenario}")
    print(f"verdict={verdict_text_multi(results)}")
    for scenario, scenario_data in results.items():
        print(f"[{scenario}]")
        for mode, data in scenario_data.items():
            print(f"{mode}: accuracy={data['accuracy']:.2f} mrr={data['mrr']:.2f}")
            for row in data["rows"]:
                status = "OK" if row["correct"] else "MISS"
                print(
                    f"  {row['step']}. expected={row['expected_title']} top={row['top_title']} "
                    f"rank={row['expected_rank']} score={row['top_score']:.2f} {status}"
                )


if __name__ == "__main__":
    main()
