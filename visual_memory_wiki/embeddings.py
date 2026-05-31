from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

from visual_memory_wiki.docs import STOPWORDS
from visual_memory_wiki.models import Node


class TextEncoder(Protocol):
    name: str

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        ...


class TfidfEncoder:
    name = "tfidf"

    def __init__(self, fit_corpus: list[str]):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOPWORDS),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]{2,}\b",
            ngram_range=(1, 2),
            max_features=1200,
        )
        self.vectorizer.fit(fit_corpus)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray()


class ClipEncoder:
    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer

        self.name = f"clip:{model_name}"
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        images = [Image.open(path).convert("RGB") for path in paths]
        return np.asarray(self.model.encode(images, normalize_embeddings=True), dtype=np.float32)


def node_texts(nodes: list[Node]) -> list[str]:
    return [f"{node.title}\n{' '.join(node.keywords)}\n{node.prompt}\n{node.text}" for node in nodes]
