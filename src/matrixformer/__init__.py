"""Self-contained inference for the frozen MatrixFormer models."""

from .artifacts import download_checkpoints
from .imputer import MatrixFormer, MatrixFormerImpute, MatrixFormerPlus, MatrixFormerRecSys

__all__ = [
    "MatrixFormer", "MatrixFormerImpute", "MatrixFormerRecSys", "MatrixFormerPlus",
    "download_checkpoints",
]
__version__ = "0.2.0"
