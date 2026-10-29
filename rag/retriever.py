import numpy as np
from .embeddings import get_embedder
from .ingestion import load_corpus, make_docs


class Retriever:
    """In-memory cosine-similarity store. Swap for FAISS / Chroma / Qdrant behind the same two methods."""

    def __init__(self, kb_dir=None, embedder="auto"):
        self.kind, self.docs, self.M = embedder, [], None
        if kb_dir:
            self.docs = load_corpus(kb_dir)
        self._build()

    def _build(self):
        if not self.docs:
            return
        self.emb = get_embedder(self.kind)
        self.emb.fit([d["text"] for d in self.docs])
        self.M = self.emb.transform([d["text"] for d in self.docs])

    def add_document(self, name, text):
        new = make_docs(name, text)
        self.docs.extend(new)
        self._build()
        return len(new)

    def search(self, query, k=4):
        if self.M is None:
            return []
        q = self.emb.transform([query])[0]
        sims = self.M @ q
        order = np.argsort(-sims)[:k]
        return [{**self.docs[i], "score": float(sims[i])} for i in order if sims[i] > 0]
