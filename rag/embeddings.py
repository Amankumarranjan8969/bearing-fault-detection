"""Embedders: sentence-transformers when installed, otherwise offline TF-IDF (no downloads needed)."""
import numpy as np


class TfidfEmbedder:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")

    def fit(self, texts):
        self.v.fit(texts)
        return self

    def transform(self, texts):
        return self.v.transform(texts).toarray().astype(np.float32)   # rows are L2-normalised


class STEmbedder:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(name)

    def fit(self, texts):
        return self

    def transform(self, texts):
        return np.asarray(self.m.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(kind="auto"):
    if kind in ("auto", "st"):
        try:
            return STEmbedder()
        except Exception:
            if kind == "st":
                raise
    return TfidfEmbedder()
