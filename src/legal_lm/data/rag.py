"""法条 RAG 检索器。

依赖 FAISS + sentence-transformers (bge-base-zh-v1.5)。
索引由 scripts/build_statute_index.py 产出，包含：
  - <output_dir>/index.faiss
  - <output_dir>/statute_meta.jsonl   每行 {"id", "law", "article", "text"}
  - <output_dir>/manifest.json        {"embed_model", "chunk_size", "dim"}

被 SFT 数据构造和推理时调用，注入 `【参考法条】` 段。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass
class StatuteChunk:
    chunk_id: int
    law: str
    article: str
    text: str

    def render(self) -> str:
        prefix = f"《{self.law}》" if self.law else ""
        if self.article:
            prefix = f"{prefix}{self.article}"
        return f"{prefix} {self.text}".strip()


def _ensure_dependencies() -> None:
    if faiss is None:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu")
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        )


class StatuteRetriever:
    """加载已构建的 FAISS 索引并支持 top-k 法条检索。"""

    def __init__(self, index_dir: str | Path, device: str | None = None) -> None:
        _ensure_dependencies()
        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Statute index manifest missing: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.embed_model_name: str = self.manifest["embed_model"]
        self.dim: int = int(self.manifest["dim"])
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.chunks: list[StatuteChunk] = []
        with (index_dir / "statute_meta.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.chunks.append(
                    StatuteChunk(
                        chunk_id=int(row["id"]),
                        law=str(row.get("law", "")),
                        article=str(row.get("article", "")),
                        text=str(row.get("text", "")),
                    )
                )
        self._encoder = SentenceTransformer(self.embed_model_name, device=device)

    def encode(self, queries: Sequence[str]):
        return self._encoder.encode(
            list(queries),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def search(self, query: str, top_k: int = 5) -> list[StatuteChunk]:
        if not query.strip():
            return []
        embedding = self.encode([query])
        scores, indices = self.index.search(embedding, top_k)
        out: list[StatuteChunk] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.chunks):
                continue
            out.append(self.chunks[int(idx)])
        return out

    def render_block(self, query: str, top_k: int = 5, max_chars: int = 1200) -> str:
        chunks = self.search(query, top_k=top_k)
        if not chunks:
            return ""
        lines = [chunk.render() for chunk in chunks]
        block = "\n".join(f"- {line}" for line in lines)
        if len(block) > max_chars:
            block = block[:max_chars].rstrip() + "..."
        return f"【参考法条】\n{block}"


def build_index(
    chunks: Iterable[StatuteChunk],
    output_dir: str | Path,
    embed_model: str = "BAAI/bge-base-zh-v1.5",
    batch_size: int = 64,
) -> None:
    _ensure_dependencies()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("No statute chunks provided.")
    encoder = SentenceTransformer(embed_model)
    embeddings = encoder.encode(
        [chunk.render() for chunk in chunk_list],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with (output_dir / "statute_meta.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunk_list:
            handle.write(
                json.dumps(
                    {
                        "id": chunk.chunk_id,
                        "law": chunk.law,
                        "article": chunk.article,
                        "text": chunk.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"embed_model": embed_model, "dim": dim, "num_chunks": len(chunk_list)},
            handle,
            ensure_ascii=False,
            indent=2,
        )


__all__ = ["StatuteChunk", "StatuteRetriever", "build_index"]
