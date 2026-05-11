"""Backport of the wandb-style API on top of swanlab.

swanlab <= 0.6 shipped `swanlab.integration.wandb` with a wandb-compatible
shim. 0.7+ removed it. The scripts in this repo were written against that
old API, and rewriting every one to use swanlab.* directly is churn. This
module exposes the *minimal* surface those scripts need:

    from stages._swanlab_shim import wandb
    wandb.init(project=..., name=..., tags=..., config=...)
    wandb.log({"metric": value}, step=step)
    if wandb.run is not None: ...
    wandb.finish()

It is intentionally tiny — anything not used in this repo is left out.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class _WandbShim:
    """Subset of the wandb module used in this repo, routed to swanlab."""

    def init(self, *args: Any, **kwargs: Any):
        try:
            import swanlab
        except ImportError:
            return None
        if args:
            raise TypeError("wandb.init expects keyword arguments only in this shim")
        kw = dict(kwargs)
        name = kw.pop("name", None)
        entity = kw.pop("entity", None)
        if name is not None:
            kw.setdefault("experiment_name", name)
        if entity is not None:
            kw.setdefault("workspace", entity)
        return swanlab.init(**kw)

    @property
    def run(self):
        try:
            import swanlab
            return swanlab.get_run()
        except Exception:
            return None

    def log(self, data: Dict[str, Any], step: Optional[int] = None, **_: Any):
        try:
            import swanlab
            return swanlab.log(data=data, step=step)
        except Exception:
            return None

    def finish(self, *_: Any, **__: Any):
        try:
            import swanlab
            return swanlab.finish()
        except Exception:
            return None


wandb = _WandbShim()
__all__ = ["wandb"]
