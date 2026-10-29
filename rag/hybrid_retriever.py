"""Hybrid RAG Retriever with BM25 Lexical Analysis, Dense Embeddings,
NLP Typo-Tolerant Query Normalizer, and Reciprocal Rank Fusion (RRF).
Handles typos, misspelled words, phonetic slang, and domain-specific terminology.
"""

import math
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

from .embeddings import get_embedder
from .ingestion import load_corpus, make_docs


# ==============================================================================
# Domain Vocabulary & Typo Correction Dictionary
# ==============================================================================
DOMAIN_DICTIONARY = {
    # Core concepts
    "bearing": ["bearin", "baring", "bairing", "bearng", "brg", "bering"],
    "vibration": ["vibrashun", "vibrtion", "vibes", "vibe", "vib", "vibrating", "vibratn"],
    "frequency": ["freq", "frequncy", "freqency", "frequecy", "frequence"],
    "outer_race": ["bpfo", "bpffo", "outer", "outerrace", "outter", "cup", "stationary_race"],
    "inner_race": ["bpfi", "bpffi", "inner", "innerrace", "cone", "shaft_race"],
    "ball": ["bsf", "rolling_element", "roller", "balls", "sphere", "rolling"],
    "cage": ["ftf", "separator", "retainer", "train"],
    "kurtosis": ["kurt", "curtosis", "kurtos", "spikiness"],
    "envelope": ["envlop", "envelop", "hilbert", "demodulation", "demod"],
    "lubrication": ["grease", "greas", "lube", "lub", "lubricat", "oil", "relubricate"],
    "maintenance": ["maintanence", "maintainance", "maint", "fix", "repair", "replace"],
    "clearance": ["clerance", "clearence", "play", "radial_clearance", "looseness"],
    "misalignment": ["misaligned", "misalignd", "angular_misalignment", "offset"],
    "resonance": ["resonanse", "natural_frequency", "harmonics", "harmonic"],
    "degradation": ["degredation", "wear", "deterioration", "failing", "broken", "damage"],
    "temperature": ["temp", "overheating", "hot", "thermal"],
    "severity": ["severty", "critical", "danger", "warning", "alarm"],
    "prognostics": ["rul", "remaining_life", "forecast", "time_to_failure", "lifetime"]
}

# Reverse lookup for O(1) typo resolution
TYPO_LOOKUP = {}
for correct, typos in DOMAIN_DICTIONARY.items():
    TYPO_LOOKUP[correct.lower()] = correct
    for typo in typos:
        TYPO_LOOKUP[typo.lower()] = correct


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes minimum edit operations between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class NLPTypoNormalizer:
    """Preprocesses colloquial human language, catches typos, and expands domain queries."""

    @classmethod
    def normalize_and_expand(cls, query: str) -> Tuple[str, List[str]]:
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+", query.lower())
        corrected_tokens = []
        corrections_made = []

        for token in raw_tokens:
            # Direct match
            if token in TYPO_LOOKUP:
                canonical = TYPO_LOOKUP[token]
                corrected_tokens.append(canonical.replace("_", " "))
                if token != canonical and token not in canonical:
                    corrections_made.append(f"'{token}' -> '{canonical}'")
                continue

            # Fuzzy match for tokens with length >= 4
            best_match = None
            best_dist = 999
            if len(token) >= 4:
                for candidate in TYPO_LOOKUP:
                    dist = levenshtein_distance(token, candidate)
                    if dist <= 2 and dist < best_dist:
                        best_dist = dist
                        best_match = TYPO_LOOKUP[candidate]

            if best_match and best_dist <= 2:
                corrected_tokens.append(best_match.replace("_", " "))
                corrections_made.append(f"'{token}' -> '{best_match}'")
            else:
                corrected_tokens.append(token)

        # Domain Query Expansion
        expanded = list(corrected_tokens)
        query_text = " ".join(corrected_tokens)
        
        if "bpfo" in query_text or "outer" in query_text:
            expanded.extend(["outer race defect", "ball pass frequency outer", "stationary impact"])
        if "bpfi" in query_text or "inner" in query_text:
            expanded.extend(["inner race defect", "ball pass frequency inner", "amplitude modulation sidebands"])
        if "bsf" in query_text or "ball" in query_text or "roller" in query_text:
            expanded.extend(["rolling element fault", "ball spin frequency", "cage modulation"])
        if "noise" in query_text or "sound" in query_text or "squeal" in query_text:
            expanded.extend(["vibration", "impact", "acoustic resonance", "kurtosis spike"])
        if "fix" in query_text or "what to do" in query_text or "action" in query_text or "replace" in query_text:
            expanded.extend(["maintenance checklist", "relubrication", "inspection procedure", "shaft alignment"])

        return " ".join(expanded), corrections_made


# ==============================================================================
# BM25 Lexical Analysis Implementation
# ==============================================================================
class BM25Okapi:
    """Standard Okapi BM25 ranking algorithm for technical lexical matching."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 1.0

        # Term frequency per doc and document frequency
        self.doc_freqs = Counter()
        self.term_freqs = []

        for doc in corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freqs[term] += 1

        # Precompute Inverse Document Frequency (IDF)
        self.idf = {}
        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formulation with smoothing
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for doc_idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                scores[doc_idx] += idf_val * (numerator / denominator)
        return scores


# ==============================================================================
# Hybrid Retriever (BM25 + Dense Semantic Embeddings + RRF)
# ==============================================================================
class HybridRetriever:
    """Industrial Hybrid Retriever combining Lexical BM25 and Dense Semantic Vectors."""

    def __init__(self, kb_dir=None, embedder="auto"):
        self.kind = embedder
        self.docs = []
        self.bm25 = None
        self.emb = None
        self.dense_matrix = None
        
        if kb_dir:
            self.docs = load_corpus(kb_dir)
        self._build()

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"[a-zA-Z0-9_]+", text) if len(w) > 1]

    def _build(self):
        if not self.docs:
            return
        
        # 1. Build BM25 Index
        corpus_tokens = [self._tokenize(d["text"]) for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens, k1=1.5, b=0.75)

        # 2. Build Dense Semantic Embedding Index
        self.emb = get_embedder(self.kind)
        texts = [d["text"] for d in self.docs]
        self.emb.fit(texts)
        self.dense_matrix = self.emb.transform(texts)

    def add_document(self, name: str, text: str) -> int:
        new_docs = make_docs(name, text)
        self.docs.extend(new_docs)
        self._build()
        return len(new_docs)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.docs or self.bm25 is None or self.dense_matrix is None:
            return []

        # 1. Normalize query with NLP & typo correction
        normalized_query, corrections = NLPTypoNormalizer.normalize_and_expand(query)
        q_tokens = self._tokenize(normalized_query)

        # 2. Compute BM25 Lexical Scores
        bm25_scores = self.bm25.score(q_tokens)
        bm25_ranks = np.argsort(-bm25_scores)

        # 3. Compute Dense Vector Cosine Similarity
        dense_q = self.emb.transform([normalized_query])[0]
        dense_scores = self.dense_matrix @ dense_q
        dense_ranks = np.argsort(-dense_scores)

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF_Score = 1 / (60 + r_bm25) + 1 / (60 + r_dense)
        rrf_scores = np.zeros(len(self.docs), dtype=np.float32)
        for rank, doc_idx in enumerate(bm25_ranks):
            if bm25_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1.0 / (60.0 + rank + 1.0)
                
        for rank, doc_idx in enumerate(dense_ranks):
            if dense_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1.0 / (60.0 + rank + 1.0)

        # Get top-k indices
        top_indices = np.argsort(-rrf_scores)[:k]
        
        results = []
        for i in top_indices:
            if rrf_scores[i] > 0:
                results.append({
                    **self.docs[i],
                    "score": float(rrf_scores[i]),
                    "bm25_score": float(bm25_scores[i]),
                    "dense_score": float(dense_scores[i]),
                    "corrections": corrections
                })

        return results
