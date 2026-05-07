from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def dump_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
