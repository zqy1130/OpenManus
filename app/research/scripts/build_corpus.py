"""Build the local retrieval corpus: fetch web documents into data/research/documents/.

Usage: python -m app.research.scripts.build_corpus

Wikipedia pages are fetched with a browser User-Agent (plain curl gets
blocked). Fetched files are raw HTML so the document loader exercises the
HTML parsing path. A manifest.json records doc ids, titles and source URLs.
"""

import json
import re
import time
from pathlib import Path

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)

DOCUMENTS = [
    "Transformer (machine learning model)",
    "Attention Is All You Need",
    "Hopfield network",
    "Boltzmann machine",
    "Geoffrey Hinton",
    "Emergent abilities of large language models",
    "Reinforcement learning",
    "GPT-4",
    "Gemini (language model)",
    "AlphaFold",
    "Turing Award",
    "Andrew Barto",
    "Richard S. Sutton",
    "DeepSeek",
    "LLaMA",
    "Cohere",
    "Character.ai",
    "杰弗里·辛顿",
    "霍普菲尔德神经网络",
    "涌现",
]

ZH_WIKI_TITLES = {"杰弗里·辛顿", "霍普菲尔德神经网络", "涌现"}


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w]+", "_", title).strip("_").lower()
    return slug or "doc"


def fetch(title: str) -> tuple[str, str]:
    base = "https://zh.wikipedia.org" if title in ZH_WIKI_TITLES else "https://en.wikipedia.org"
    url = f"{base}/wiki/{title.replace(' ', '_')}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
    response.raise_for_status()
    return url, response.text


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "research" / "documents"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, title in enumerate(DOCUMENTS, 1):
        file = out_dir / f"{i:02d}_{slugify(title)}.html"
        if file.exists():
            url = (
                f"https://{'zh' if title in ZH_WIKI_TITLES else 'en'}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )
            html = ""
            print(f"[{i}/{len(DOCUMENTS)}] {slugify(title)}: exists, skipping")
        else:
            url, html = fetch(title)
            file.write_text(html, encoding="utf-8")
            print(f"[{i}/{len(DOCUMENTS)}] {slugify(title)}: {len(html)} bytes")
            time.sleep(1)
        doc_id = slugify(title)
        manifest.append(
            {
                "doc_id": doc_id,
                "title": title,
                "url": url,
                "file": file.name,
            }
        )

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(manifest)} documents to {out_dir}")


if __name__ == "__main__":
    main()
