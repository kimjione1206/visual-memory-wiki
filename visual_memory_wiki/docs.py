from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from visual_memory_wiki.models import Node


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


def clean_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text.strip()


def title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def extract_keywords(docs: list[str], top_k: int = 6) -> list[list[str]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(STOPWORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
        ngram_range=(1, 2),
        max_features=900,
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
            if len(terms) >= top_k:
                break
        result.append(terms)
    return result


def load_markdown_nodes(docs_dir: Path, top_k: int = 6) -> list[Node]:
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files found in {docs_dir}")
    raw_texts = [path.read_text(encoding="utf-8") for path in files]
    clean_texts = [clean_markdown(text) for text in raw_texts]
    keyword_sets = extract_keywords(clean_texts, top_k=top_k)
    nodes: list[Node] = []
    for idx, (path, raw, clean, keywords) in enumerate(zip(files, raw_texts, clean_texts, keyword_sets), start=1):
        title = title_from_markdown(raw, path.stem)
        prompt = f"{title}. Visual metaphor: {', '.join(keywords[:5])}. {clean[:240]}"
        nodes.append(
            Node(
                id=f"n{idx:02d}",
                title=title,
                source_path=path.resolve(),
                text=clean,
                keywords=keywords,
                prompt=prompt,
            )
        )
    return nodes
