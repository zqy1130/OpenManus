"""Unit tests for app.research.bm25."""

from app.research.bm25 import BM25Index, tokenize


class TestTokenize:
    def test_latin_words_lowercased(self):
        assert tokenize("Hello World 42") == ["hello", "world", "42"]

    def test_cjk_bigrams(self):
        tokens = tokenize("神经网络")
        assert "神经" in tokens and "经网" in tokens and "网络" in tokens

    def test_single_cjk_char(self):
        assert tokenize("网") == ["网"]

    def test_mixed(self):
        tokens = tokenize("Hopfield 网络")
        assert "hopfield" in tokens
        assert "网络" in tokens


class TestBM25Index:
    def build(self):
        index = BM25Index()
        index.build(
            [
                {"id": "d1", "text": "the cat sat on the mat with the cat toy"},
                {"id": "d2", "text": "the dog ran in the park chasing a ball"},
            ]
        )
        return index

    def test_relevant_doc_ranks_first(self):
        index = self.build()
        hits = index.search("cat", top_k=2)
        assert hits[0][0] == "d1"
        assert hits[0][1] > 0

    def test_top_k_limit(self):
        index = self.build()
        hits = index.search("the", top_k=1)
        assert len(hits) == 1

    def test_empty_query(self):
        index = self.build()
        assert index.search("", top_k=5) == []

    def test_unknown_terms_return_empty(self):
        index = self.build()
        assert index.search("zzzznotaword", top_k=5) == []

    def test_cjk_query(self):
        index = BM25Index()
        index.build(
            [
                {"id": "zh1", "text": "霍普菲尔德网络是一种联想记忆网络"},
                {"id": "en1", "text": "the quick brown fox jumps over the lazy dog"},
            ]
        )
        hits = index.search("霍普菲尔德网络", top_k=1)
        assert hits and hits[0][0] == "zh1"
