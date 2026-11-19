"""Self-contained inference for the single pretrained MatrixFormer model."""

from .artifacts import download_checkpoints
from .imputer import MatrixFormer

__all__ = ["MatrixFormer", "download_checkpoints"]
__version__ = "0.3.0"
