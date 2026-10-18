"""Self-contained inference for the frozen MatrixFormer experts."""

from .artifacts import download_checkpoints
from .imputer import MatrixFormerImpute, MatrixFormerPlus, MatrixFormerRecSys

__all__ = [
    "MatrixFormerImpute", "MatrixFormerPlus", "MatrixFormerRecSys", "download_checkpoints",
]
__version__ = "0.1.0"
