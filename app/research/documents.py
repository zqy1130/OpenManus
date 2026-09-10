"""Local document loading and chunking for the research corpus.

Supports HTML (BeautifulSoup, strips nav/scripts) and Markdown/plain text.
Documents are chunked greedily by paragraphs with a target size, keeping
chunk boundaries at paragraph and sentence boundaries.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from bs4 import BeautifulSoup

DEFAULT_CORPUS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "research" / "documents"
)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    section: str = ""
    index: int = 0


@dataclass
class Document:
    doc_id: str
    title: str
    url: str = ""
    text: str = ""
    chunks: List[Chunk] = field(default_factory=list)


def html_to_text(html: str) -> tuple[str, str]:
    """Extract (title, body text) from an HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(
        ["script", "style", "nav", "header", "footer", "noscript", "aside"]
    ):
        tag.extract()
    content = soup.find("div", id="mw-content-text") or soup.body or soup
    text = content.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def chunk_text(
    text: str,
    doc_id: str,
    target_chars: int = 500,
    overlap_chars: int = 50,
    section: str = "",
) -> List[Chunk]:
    """Split text into chunks of ~target_chars, preserving paragraph and
    sentence boundaries. Long sentences are hard-split when necessary."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[Chunk] = []
    buffer = ""
    index = 0

    def flush() -> None:
        nonlocal buffer, index
        if buffer.strip():
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{index:03d}",
                    text=buffer.strip(),
                    section=section,
                    index=index,
                )
            )
            index += 1
            buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) > target_chars * 2:
            pieces = re.split(r"(?<=[.!?。！？])\s+", paragraph)
            for piece in pieces:
                if len(piece) > target_chars * 2:
                    for j in range(0, len(piece), target_chars):
                        buffer += piece[j : j + target_chars] + "\n"
                        flush()
                elif len(buffer) + len(piece) > target_chars:
                    flush()
                    buffer = piece + "\n"
                else:
                    buffer += piece + "\n"
            continue
        if len(buffer) + len(paragraph) > target_chars:
            flush()
            buffer = paragraph + "\n"
        else:
            buffer += paragraph + "\n"
    flush()

    if overlap_chars and len(chunks) > 1:
        for prev, cur in zip(chunks, chunks[1:]):
            tail = prev.text[-overlap_chars:]
            cur.text = tail + " " + cur.text
    return chunks


def load_document(path: Path, doc_id: Optional[str] = None) -> Document:
    """Load one .html/.md/.txt file into a Document (text only, unchunked)."""
    if doc_id is None:
        doc_id = path.stem
    if path.suffix == ".html":
        title, text = html_to_text(path.read_text(encoding="utf-8", errors="replace"))
        return Document(doc_id=doc_id, title=title, text=text)
    return Document(doc_id=doc_id, title=path.stem, text=path.read_text(encoding="utf-8"))


def load_corpus(
    corpus_dir: Optional[Path] = None,
    target_chars: int = 500,
    overlap_chars: int = 50,
) -> List[Document]:
    """Load every document in the corpus dir and chunk them.

    The manifest.json produced by scripts/build_corpus.py supplies doc ids,
    titles and source URLs; files without a manifest entry fall back to
    their file stem.
    """
    corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
    manifest_path = corpus_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        for entry in json.loads(manifest_path.read_text(encoding="utf-8")):
            manifest[entry["file"]] = entry

    documents: List[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix not in (".html", ".md", ".txt"):
            continue
        entry = manifest.get(path.name, {})
        doc = load_document(path, doc_id=entry.get("doc_id"))
        doc.url = entry.get("url", "")
        if entry.get("title"):
            doc.title = entry["title"]
        doc.chunks = chunk_text(
            doc.text,
            doc.doc_id,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        documents.append(doc)
    return documents
