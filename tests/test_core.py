from pathlib import Path

from visual_memory_wiki.cards import generate_cards
from visual_memory_wiki.docs import load_markdown_nodes
from visual_memory_wiki.embeddings import TfidfEncoder, node_texts
from visual_memory_wiki.graphing import build_similarity_graph
from visual_memory_wiki.retrieval import DEFAULT_CUES, make_walk


def test_load_markdown_nodes_and_generate_cards(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Alpha\n\nImages act as visual memory links.", encoding="utf-8")
    (docs / "b.md").write_text("# Beta\n\nGraphs connect cards and source notes.", encoding="utf-8")

    nodes = load_markdown_nodes(docs)
    assert [node.title for node in nodes] == ["Alpha", "Beta"]

    card_nodes = generate_cards(nodes, tmp_path / "cards", style="textless")
    assert all(node.image_path and node.image_path.exists() for node in card_nodes)


def test_tfidf_walk_smoke(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# Upload\n\nUploaded notes become keyword image cards.", encoding="utf-8")
    (docs / "b.md").write_text("# Walker\n\nAn AI walker reads nearby useful cards.", encoding="utf-8")

    nodes = load_markdown_nodes(docs)
    encoder = TfidfEncoder(node_texts(nodes) + DEFAULT_CUES[:2])
    node_matrix = encoder.encode_texts(node_texts(nodes))
    cue_matrix = encoder.encode_texts(DEFAULT_CUES[:2])
    graph, _ = build_similarity_graph(nodes, node_matrix, threshold=0.0)
    walk = make_walk(nodes, DEFAULT_CUES[:2], cue_matrix, node_matrix)

    assert graph.number_of_nodes() == 2
    assert len(walk) == 2
    assert walk[0].node_id in {"n01", "n02"}
