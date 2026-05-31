from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from visual_memory_wiki.cards import attach_generated_cards, generate_cards
from visual_memory_wiki.docs import load_markdown_nodes
from visual_memory_wiki.embeddings import ClipEncoder, TfidfEncoder, node_texts
from visual_memory_wiki.exporters import export_canvas, export_html, export_obsidian_notes, write_json_report
from visual_memory_wiki.graphing import build_similarity_graph
from visual_memory_wiki.retrieval import DEFAULT_CUES, make_walk, rank_cues


SAMPLE_EXPECTED = {
    1: "n02",
    2: "n01",
    3: "n03",
    4: "n04",
    5: "n07",
    6: "n08",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visual-wiki",
        description="Image-based knowledge exploration experiment toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build an HTML viewer and Obsidian Canvas from markdown notes.")
    build.add_argument("--docs", type=Path, default=Path("demo_knowledge"))
    build.add_argument("--out", type=Path, default=Path("dist/visual-wiki"))
    build.add_argument("--backend", choices=["tfidf", "clip", "auto"], default="auto")
    build.add_argument("--card-style", choices=["text", "textless"], default="textless")
    build.add_argument("--generated-card-dir", type=Path)
    build.add_argument("--clip-model", default="sentence-transformers/clip-ViT-B-32")
    build.add_argument("--title", default="Visual Memory Wiki")
    build.add_argument("--query", default="Image-based knowledge exploration experiment.")

    evaluate = sub.add_parser("eval", help="Compare retrieval modes on the bundled demo cues.")
    evaluate.add_argument("--docs", type=Path, default=Path("demo_knowledge"))
    evaluate.add_argument("--out", type=Path, default=Path("dist/eval"))
    evaluate.add_argument("--generated-card-dir", type=Path)
    evaluate.add_argument("--clip-model", default="sentence-transformers/clip-ViT-B-32")
    evaluate.add_argument("--no-clip", action="store_true", help="Skip CLIP modes and run only TF-IDF.")
    return parser


def _load_cues() -> list[str]:
    return DEFAULT_CUES


def _attach_cards(nodes, out: Path, style: str, generated_card_dir: Path | None = None):
    if generated_card_dir:
        return attach_generated_cards(nodes, generated_card_dir, out / "generated_cards")
    return generate_cards(nodes, out / f"{style}_cards", style=style)


def _build_with_tfidf(nodes, cues):
    corpus = node_texts(nodes) + cues
    encoder = TfidfEncoder(corpus)
    node_matrix = encoder.encode_texts(node_texts(nodes))
    cue_matrix = encoder.encode_texts(cues)
    return encoder.name, node_matrix, cue_matrix


def _build_with_clip(nodes, cues, model_name: str):
    encoder = ClipEncoder(model_name)
    image_paths = [node.image_path for node in nodes]
    if any(path is None for path in image_paths):
        raise ValueError("CLIP backend requires image cards.")
    node_matrix = encoder.encode_images([Path(path) for path in image_paths if path is not None])
    cue_matrix = encoder.encode_texts(cues)
    return encoder.name, node_matrix, cue_matrix


def run_build(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    nodes = load_markdown_nodes(args.docs)
    nodes = _attach_cards(nodes, out, args.card_style, args.generated_card_dir)
    cues = _load_cues()

    if args.backend == "tfidf":
        backend, node_matrix, cue_matrix = _build_with_tfidf(nodes, cues)
    else:
        try:
            backend, node_matrix, cue_matrix = _build_with_clip(nodes, cues, args.clip_model)
        except Exception as exc:
            if args.backend == "clip":
                raise
            print(f"CLIP unavailable, falling back to TF-IDF: {type(exc).__name__}: {exc}")
            backend, node_matrix, cue_matrix = _build_with_tfidf(nodes, cues)

    graph, _ = build_similarity_graph(nodes, node_matrix)
    walk = make_walk(nodes, cues, cue_matrix, node_matrix)
    html_path = export_html(nodes, graph, walk, out, title=args.title)
    canvas_path = export_canvas(nodes, graph, walk, out)
    obsidian_note = export_obsidian_notes(nodes, walk, out, query=args.query)
    report_path = write_json_report(
        out / "run_report.json",
        {
            "backend": backend,
            "documents": len(nodes),
            "images": len(nodes),
            "edges": graph.number_of_edges(),
            "html": html_path,
            "canvas": canvas_path,
            "obsidian_walk_note": obsidian_note,
            "walk": walk,
            "note": "This is an experimental visual retrieval run. Image cards are indexes; source notes remain the ground truth.",
        },
    )
    print(f"backend={backend}")
    print(f"html={html_path}")
    print(f"canvas={canvas_path}")
    print(f"report={report_path}")


def run_eval(args: argparse.Namespace) -> None:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_nodes = load_markdown_nodes(args.docs)
    cues = _load_cues()
    results = {}

    backend, node_matrix, cue_matrix = _build_with_tfidf(base_nodes, cues)
    results["text_baseline"] = rank_cues(base_nodes, cues, cue_matrix, node_matrix, expected=SAMPLE_EXPECTED)
    results["text_baseline"]["backend"] = backend

    if not args.no_clip:
        text_nodes = generate_cards(base_nodes, out / "text_cards", style="text")
        textless_nodes = generate_cards(base_nodes, out / "textless_cards", style="textless")
        for mode, nodes in [("text_card_clip", text_nodes), ("textless_card_clip", textless_nodes)]:
            backend, node_matrix, cue_matrix = _build_with_clip(nodes, cues, args.clip_model)
            results[mode] = rank_cues(nodes, cues, cue_matrix, node_matrix, expected=SAMPLE_EXPECTED)
            results[mode]["backend"] = backend
        if args.generated_card_dir:
            generated_nodes = attach_generated_cards(base_nodes, args.generated_card_dir, out / "generated_cards")
            backend, node_matrix, cue_matrix = _build_with_clip(generated_nodes, cues, args.clip_model)
            results["generated_card_clip"] = rank_cues(generated_nodes, cues, cue_matrix, node_matrix, expected=SAMPLE_EXPECTED)
            results["generated_card_clip"]["backend"] = backend

    serializable = {
        mode: {
            **{k: v for k, v in result.items() if k != "rows"},
            "rows": [asdict(row) for row in result["rows"]],
        }
        for mode, result in results.items()
    }
    report_path = write_json_report(out / "eval_report.json", {"results": serializable})
    for mode, result in results.items():
        print(f"{mode}: accuracy={result['accuracy']} mrr={result['mrr']}")
    print(f"report={report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        run_build(args)
    elif args.command == "eval":
        run_eval(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
