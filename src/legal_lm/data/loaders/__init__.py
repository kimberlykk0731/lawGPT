"""Dataset loaders for unified legal schema."""

from .base import BaseDatasetLoader
from .cail import CAILLoader
from .jec_qa import JECQALoader
from .lecard_v2 import LeCaRDv2Loader
from .local_jsonl import LocalJsonlLoader

__all__ = [
    "BaseDatasetLoader",
    "CAILLoader",
    "JECQALoader",
    "LeCaRDv2Loader",
    "LocalJsonlLoader",
]
