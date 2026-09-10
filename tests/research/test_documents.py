"""Unit tests for app.research.documents."""

import json
from pathlib import Path

from app.research.documents import chunk_text, html_to_text, load_corpus, load_document

SAMPLE_HTML = """<html><head><title>Test Page</title></head>
<body><div id="mw-content-text">
<h2>Section One</h2>
<p>First paragraph with enough text to form a chunk on its own.
It keeps going with more sentences to exceed the target size limit.
And yet another sentence here to be safe about length.</p>
<h2>Section Two</h2>
<p>Second paragraph content.</p>
</div></body></html>"""


class TestHtmlToText:
    def test_extracts_title_and_body(self):
        title, text = html_to_text(SAMPLE_HTML)
        assert title == "Test Page"
        assert "First paragraph" in text
        assert "Second paragraph" in text
        assert "Section One" in text

    def test_strips_scripts(self):
        html = "<html><head><title>t</title></head><body><script>evil()</script><p>good</p></body></html>"
        _, text = html_to_text(html)
        assert "evil" not in text
        assert "good" in text


class TestChunkText:
    def test_chunks_have_ids_and_content(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, doc_id="d1", target_chars=12)
        assert len(chunks) >= 1
        assert all(c.chunk_id.startswith("d1#") for c in chunks)
        assert "".join(c.text for c in chunks).replace("\n", " ").strip() != ""

    def test_paragraphs_are_not_split_across_chunks(self):
        long = "word " * 60  # ~300 chars
        text = f"{long}\n\n{long}\n\n{long}"
        chunks = chunk_text(text, doc_id="d", target_chars=350)
        # each paragraph is ~300 chars < 350, so each chunk holds one paragraph
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.text.count("\n") == 0

    def test_overlap_is_added_between_chunks(self):
        text = ("sentence one. " * 20) + "\n\n" + ("sentence two. " * 20)
        chunks = chunk_text(text, doc_id="d", target_chars=100, overlap_chars=20)
        assert len(chunks) >= 2
        assert chunks[1].text.startswith(chunks[0].text[-20:])


class TestLoadCorpus:
    def test_loads_html_and_manifest(self, tmp_path: Path):
        (tmp_path / "01_doc_a.html").write_text(SAMPLE_HTML, encoding="utf-8")
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "doc_id": "doc_a",
                        "title": "Doc A Title",
                        "url": "https://example.com/a",
                        "file": "01_doc_a.html",
                    }
                ]
            ),
            encoding="utf-8",
        )
        docs = load_corpus(tmp_path)
        assert len(docs) == 1
        assert docs[0].doc_id == "doc_a"
        assert docs[0].title == "Doc A Title"
        assert docs[0].url == "https://example.com/a"
        assert len(docs[0].chunks) >= 1

    def test_load_document_markdown(self, tmp_path: Path):
        path = tmp_path / "note.md"
        path.write_text("# Hello\n\nSome text.", encoding="utf-8")
        doc = load_document(path)
        assert doc.title == "note"
        assert "Some text" in doc.text
