"""Dependency-free BM25 implementation with CJK-aware tokenization.

Latin text is tokenized into lowercase word tokens; CJK runs are tokenized
into character bigrams (a standard approach when no segmenter is available),
which gives BM25 reasonable recall for Chinese text without jieba.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[一-鿿]+")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_PATTERN.finditer(text.lower()):
        token = match.group()
        if token[0].isascii():
            tokens.append(token)
        elif len(token) == 1:
            tokens.append(token)
        else:
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))
    return tokens


class BM25Index:
    """Standard BM25 (Robertson et al.) over a set of documents.

    Documents are dicts with at least 'id' and 'text' keys.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_tokens: Dict[str, List[str]] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.doc_freq: Counter = Counter()
        self.avgdl: float = 0.0

    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        if not tokens:
            return
        if doc_id not in self.doc_tokens:
            self.doc_ids.append(doc_id)
        self.doc_tokens[doc_id] = tokens
        self.doc_lengths[doc_id] = len(tokens)
        for term in set(tokens):
            self.doc_freq[term] += 1

    def build(self, docs: List[dict]) -> "BM25Index":
        for doc in docs:
            self.add(doc["id"], doc["text"])
        if self.doc_ids:
            self.avgdl = sum(self.doc_lengths.values()) / len(self.doc_ids)
        return self

    def _score(self, query_tokens: List[str], doc_id: str) -> float:
        tf = Counter(self.doc_tokens[doc_id])
        dl = self.doc_lengths[doc_id]
        score = 0.0
        for term in set(query_tokens):
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self.doc_ids) - df + 0.5) / (df + 0.5))
            numerator = tf.get(term, 0) * (self.k1 + 1)
            denominator = tf.get(term, 0) + self.k1 * (
                1 - self.b + self.b * dl / max(self.avgdl, 1.0)
            )
            score += idf * numerator / denominator
        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Return top_k (doc_id, score) pairs for a query."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scored = [
            (doc_id, self._score(query_tokens, doc_id)) for doc_id in self.doc_ids
        ]
        scored = [(doc_id, score) for doc_id, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
