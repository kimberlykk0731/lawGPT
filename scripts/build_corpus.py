"""法律语料清洗 + 脱敏。

读入 raw 目录下的 JSONL / TXT 文件，对每个文档做长度过滤、去重、可选脱敏，
统一输出 JSONL：{"doc_id", "source_file", "text", "redact_stats"}。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from legal_lm.data.redact import Redactor


def _iter_documents(input_dir: Path):
    for path in sorted(input_dir.glob("**/*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get("text") or record.get("content") or ""
                    if not isinstance(text, str):
                        continue
                    yield path, idx, text
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            yield path, 0, text


def _normalize(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="法律语料清洗 + 脱敏")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--redact", action="store_true")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=16000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "corpus.jsonl"
    seen: set[str] = set()
    total = kept = 0
    handle = out_path.open("w", encoding="utf-8")
    try:
        for path, idx, raw in _iter_documents(input_dir):
            total += 1
            text = _normalize(raw)
            if not (args.min_length <= len(text) <= args.max_length):
                continue
            digest = hashlib.md5(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            stats = {}
            if args.redact:
                redactor = Redactor()
                text = redactor.redact(text)
                stats = redactor.stats.__dict__
            handle.write(
                json.dumps(
                    {
                        "doc_id": digest,
                        "source_file": str(path.relative_to(input_dir)),
                        "source_index": idx,
                        "text": text,
                        "redact_stats": stats,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
    finally:
        handle.close()
    print(f"[build_corpus] total={total} kept={kept} -> {out_path}")


if __name__ == "__main__":
    main()
