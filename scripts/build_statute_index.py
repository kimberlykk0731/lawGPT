"""法条 RAG 向量索引构建。

输入：data/raw/statutes/ 下的 *.txt 或 *.jsonl 法条文本。
TXT 解析约定：
  《xx法》
  第N条 ...条文内容...

JSONL 解析约定：
  {"law": "刑法", "article": "第二百六十四条", "text": "..."}

输出：FAISS 索引 + statute_meta.jsonl + manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from legal_lm.data.rag import StatuteChunk, build_index


LAW_TITLE_PATTERN = re.compile(r"《([^》]+)》")
ARTICLE_HEADER_PATTERN = re.compile(r"^(第[一二三四五六七八九十百千零\d]+条)")


@dataclass
class _ParsedArticle:
    law: str
    article: str
    text: str


def _parse_txt_file(path: Path) -> list[_ParsedArticle]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    law_match = LAW_TITLE_PATTERN.search(text)
    law = law_match.group(1) if law_match else path.stem
    articles: list[_ParsedArticle] = []
    current_article = ""
    current_text: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = ARTICLE_HEADER_PATTERN.match(line)
        if match:
            if current_article and current_text:
                articles.append(
                    _ParsedArticle(law=law, article=current_article, text=" ".join(current_text))
                )
            current_article = match.group(1)
            current_text = [line[len(current_article):].strip()]
        else:
            if current_article:
                current_text.append(line)
    if current_article and current_text:
        articles.append(_ParsedArticle(law=law, article=current_article, text=" ".join(current_text)))
    return articles


def _parse_jsonl_file(path: Path) -> list[_ParsedArticle]:
    rows: list[_ParsedArticle] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(
                _ParsedArticle(
                    law=str(rec.get("law", path.stem)),
                    article=str(rec.get("article", "")),
                    text=str(rec.get("text", rec.get("content", ""))),
                )
            )
    return rows


def _chunk_article(article: _ParsedArticle, chunk_size: int) -> list[str]:
    body = article.text.strip()
    if len(body) <= chunk_size:
        return [body]
    chunks: list[str] = []
    cursor = 0
    while cursor < len(body):
        chunks.append(body[cursor : cursor + chunk_size])
        cursor += chunk_size
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="构建法条 RAG 索引")
    parser.add_argument("--statute-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embed-model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--incremental", action="store_true",
                        help="占位：当前实现总是重建索引；后续可接入差量构建")
    args = parser.parse_args()

    statute_dir = Path(args.statute_dir)
    if not statute_dir.exists():
        raise FileNotFoundError(statute_dir)

    raw_articles: list[_ParsedArticle] = []
    for path in sorted(statute_dir.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            raw_articles.extend(_parse_jsonl_file(path))
        elif suffix in {".txt", ".md"}:
            raw_articles.extend(_parse_txt_file(path))
    if not raw_articles:
        raise RuntimeError(f"No articles parsed under {statute_dir}")

    chunks: list[StatuteChunk] = []
    cid = 0
    for article in raw_articles:
        for piece in _chunk_article(article, args.chunk_size):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(StatuteChunk(chunk_id=cid, law=article.law, article=article.article, text=piece))
            cid += 1

    print(f"[build_statute_index] articles={len(raw_articles)} chunks={len(chunks)}")
    build_index(chunks, output_dir=args.output_dir, embed_model=args.embed_model)
    print(f"[build_statute_index] done -> {args.output_dir}")


if __name__ == "__main__":
    main()
