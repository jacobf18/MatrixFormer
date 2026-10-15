# MatrixFormer+

Complete numerical matrices with two frozen pretrained transformers and an
observed-error-fitted convex blend. Supply a 2D NumPy array with `np.nan` for
missing entries. The output preserves observed entries exactly.

This is a standalone, inference-only repository. All model, streaming,
normalization, and blending code lives here. It does not import `tabimpute`,
depend on the `tabular` repository, or contain datasets, training jobs, or
benchmark infrastructure.

- [Pretrained weights](https://huggingface.co/Tabimpute/TabImpute)
- [Paper source](https://github.com/jacobf18/MatrixFormerPaper)

## Install

Use Python 3.10 or newer. Install an appropriate PyTorch build for your hardware,
then install this package:

```bash
git clone https://github.com/jacobf18/MatrixFormer.git
cd MatrixFormer
python -m pip install .
```

The only runtime dependencies are PyTorch 2.4+, NumPy, and `huggingface-hub`.
Use `python -m pip install -e .` instead if editing the code.

## Python

```python
import numpy as np
from matrixformer import MatrixFormerPlus

matrix = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 6.0],
    [3.0, 6.0, 9.0],
])

model = MatrixFormerPlus.from_pretrained(device="cpu")
completed = model.impute(matrix)

print(completed)
print(model.last_decision)
```

The first call downloads about 624 MB of weights and the matching configs from
Hugging Face. Downloads use a pinned release and are checked against SHA-256
hashes before loading. The models are public; no token is required. Later calls
reuse the Hugging Face cache.

For GPU inference, use `device="cuda"` or `device="cuda:1"`. The default precision
is `amp_bfloat16`, matching the published inference recipe. Use
`precision="float32"` on hardware without BF16 support. Precision and hardware
can affect predictions.

The single general expert is also available:

```python
from matrixformer import MatrixFormerImpute

model = MatrixFormerImpute.from_pretrained(device="cpu")
completed = model.impute(matrix)
```

Loading MatrixFormer-Impute downloads only its own checkpoint, about 312 MB.
Reuse an instantiated model across matrices to avoid repeated weight loading.

## Command Line

Run a small synthetic example without preparing any input files:

```bash
python -m matrixformer
```

Complete your own `.npy` matrix, with `np.nan` marking missing entries:

```bash
matrixformer --input matrix.npy --output completed.npy --device cuda
matrixformer --model impute --input matrix.npy --output completed.npy --device cpu
```

Both commands print an inference summary. `--output` saves the completed array;
the input file is left unchanged. Use `matrixformer --help` for all options.
These examples perform inference only, including the synthetic demonstration.

## Offline Use

Download both experts into a portable directory:

```bash
matrixformer --download-only --local-dir ./checkpoints
matrixformer --local-dir ./checkpoints --input matrix.npy --output completed.npy
```

Or load that directory from Python:

```python
model = MatrixFormerPlus.from_pretrained(local_dir="./checkpoints", device="cpu")
completed = model.impute(matrix)
```

`local_dir` is read-only during inference and makes no network requests. Its
layout is compatible with the existing Hugging Face bundle:

```text
checkpoints/
  matrixformer/
    impute/
      config.json
      best_validation_checkpoint.pth
    recsys/
      config.json
      best_validation_checkpoint.pth
```

Alternatively, download to the default Hub cache with
`matrixformer --download-only`, then use `local_files_only=True` in Python or
`--local-files-only` on the command line. The installed package includes its
own architecture code and decoder-bin constants. No files from another source
checkout are needed, including when running offline.

## Context and Memory

The default context is at most 2000 rows by 1000 columns. Larger matrices use
overlapping windows. Column normalization is computed from the observed values
of the whole supplied matrix before windowing.

```python
model = MatrixFormerPlus.from_pretrained(
    device="cuda",
    max_rows=2000,
    max_cols=1000,
)
```

Use `None` for either Python limit to retain the full axis. On the command line,
`--max-rows 0 --max-cols 0` requests full context. Smaller limits reduce memory
requirements but can change predictions.

Attention batches, feed-forward cells, and decoder outputs are streamed in
microbatches without shortening attention within a context window. The hidden
grid still stays on the selected device. Large windows require substantial
memory, and both expert networks remain loaded. Start with the small CPU demo;
use a GPU for larger matrices.

## What the Blend Does

Both experts use the Dwip v2-2 architecture, with 77,998,496 parameters each.
Their weights were trained on synthetic matrices and remain frozen at inference.
MatrixFormer+ runs the general expert and the recommender specialist, then fits
one specialist weight between zero and one for the entire input matrix. The
criterion is column-standardized squared reconstruction error on observed
entries. `last_decision["specialist_weight"]` records that weight.

The fitting step uses raw model predictions on observed entries, rather than
copying their targets into the loss. Those entries are also visible to the
encoder, so this is an in-sample criterion, not a held-out accuracy estimate.
No hidden ground-truth values are supplied, no neural network parameters are
updated, and there is no separate MatrixFormer+ checkpoint.

This package implements the generic column-normalized recipe. It does not yet
include the benchmark-specific preprocessing that produced the paper's rating
tables: global normalization for the general expert, two-way centering and user
scaling for the specialist, rating clipping, and different context limits.
Loading these weights with the generic recipe does not reproduce every paper
result. Evaluation datasets and benchmark runners are intentionally excluded.

Inputs must be real numerical matrices. Encode categorical data separately.
Observed values must be exactly representable as float64; lossy integer or
extended-precision conversions are rejected rather than silently changing them.
An entirely missing matrix is rejected. Entirely missing rows or columns have
limited observational support. Extreme values, unusual missingness patterns,
or matrix structure outside the synthetic training distribution can reduce
accuracy. Predictions are means, not calibrated uncertainty estimates.

## Repository Layout

```text
README.md                  Installation, usage, and limitations
pyproject.toml             Package metadata and three runtime dependencies
src/matrixformer/
  __init__.py              Public API
  __main__.py              CLI and synthetic demo
  imputer.py               Input handling and observed-error blending
  artifacts.py             Pinned downloads and checksum verification
  _model.py                Frozen transformer architecture
  _inference.py            Streaming and predictive-mean decoding
  assets/checkpoints.json  Release hashes and provenance
  assets/borders.pt        Small decoder-bin constants, not a dataset
```

The runtime was extracted from the published `tabular` source revision
`2cf54ac47dbcc1154b35e875b527ed0c42a9f416`. Training-only paths and unused model
variants were removed. The checkpoint keys and inference arithmetic retain the
released architecture. `assets/checkpoints.json` pins Hugging Face release
`727574989ddebe71860141c9a8b9f6e7958e3cce` and identifies both original experts.

CPU validation matches the published runtime exactly on 16 synthetic cases
across BF16 autocast and float32, including window overlaps, missing axes, and
streaming boundaries. An installed wheel also passes offline inference with
network access and unrelated imports blocked. These are implementation checks,
not benchmark accuracy claims; GPU parity has not been tested in this extraction.
