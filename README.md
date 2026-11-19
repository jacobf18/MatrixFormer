# MatrixFormer

Fill missing entries in numerical matrices with a pretrained transformer.
MatrixFormer is a single network trained on a mix of synthetic imputation,
structured-missingness, and rating priors. Supply a 2D NumPy array with
`np.nan` for missing entries. The output preserves observed entries exactly.
You do not need to train or fine-tune the model. This release contains one
MatrixFormer checkpoint, about 312 MB.

This is a standalone repository. All model, streaming, normalization, and
decoding code lives here, with optional [synthetic training](TRAINING.md).
It does not import `tabimpute`, depend on the `tabular` repository, or contain
evaluation datasets or benchmark infrastructure.

- [Project page](https://jacobf18.github.io/MatrixFormer/)
- [Hugging Face weights and configs](https://huggingface.co/Tabimpute/MatrixFormer)
- [Paper](https://arxiv.org/abs/2610.06751)

## Install and First Run

You need Git, Python 3.10 or newer, and internet access for installation and the
first model download. A GPU and a Hugging Face account are optional. Allow disk
space for PyTorch and its dependencies as well as about 312 MB for the default
model.

Here is a CPU-only setup for Linux, using an isolated virtual environment:

```bash
git clone https://github.com/jacobf18/MatrixFormer.git
cd MatrixFormer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.4,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python -m matrixformer --device cpu --precision float32
```

The last command downloads the default model and fills 30 missing entries in a small
synthetic matrix. It prints a JSON summary with `shape: [12, 9]`,
`missing_entries: 30`, `all_predictions_finite: true`, and
`observed_entries_preserved: true`. It runs inference only and saves no file.
These checks confirm the installation works, rather than measure accuracy.

For NVIDIA GPUs or macOS, replace the CPU PyTorch install command with the one
from the [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install this package. Only CPU and NVIDIA CUDA inference are supported;
on macOS use CPU, not `mps`. For Windows PowerShell, create the environment with
`python -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`.
Reactivate the environment whenever you open a new terminal.

If creating the environment reports that `ensurepip` is unavailable on
Debian/Ubuntu, install the venv package for your Python version (typically
`sudo apt install python3-venv`) and retry.

The only runtime dependencies are PyTorch 2.4+, NumPy, and `huggingface-hub`.
Install from this repository with `python -m pip install .`, rather than assuming
the package is available on PyPI. Use `python -m pip install -e .` instead if
editing the code. After installation, you can run it from any directory.

## One Model

Use `MatrixFormer.from_pretrained()` or the default CLI. Weights and the minimal
config are stored directly at the root of `Tabimpute/MatrixFormer`.

## Training

To train a single MatrixFormer from random initialization using the packaged
synthetic prior:

```bash
python -m pip install -e ".[training]"
python -m matrixformer.training --output runs/matrixformer --device cuda
```

The prior combines scientific panels, spectral/latent-factor imputation tasks,
sparse ratings, and score panels. Training uses synthetic targets only, with
Huber, density, and CDF losses. The trainer includes independent synthetic
validation, resumable checkpoints, and gradient-safe microbatching. See
[TRAINING.md](TRAINING.md) for the recipe, memory controls, and a small CPU smoke
run. Training is optional; pretrained inference needs no training dependencies.

## Python

```python
import numpy as np
from matrixformer import MatrixFormer

matrix = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 6.0],
    [3.0, 6.0, 9.0],
])

model = MatrixFormer.from_pretrained(device="cpu", precision="float32")
completed = model.impute(matrix)

print(completed)
```

`completed` is a float64 NumPy array with the same shape and units as `matrix`.
The input is left unchanged. Column normalization and conversion back to the
original units happen internally. Reuse the model object for multiple matrices.

When `local_dir` is omitted, `MatrixFormer.from_pretrained()` automatically
downloads the model (about 312 MB of weights) and its matching config from
[Tabimpute/MatrixFormer on Hugging Face](https://huggingface.co/Tabimpute/MatrixFormer).
The CLI does the same by default; no model paths or separate download step are
needed. Downloads use a pinned release and are checked against SHA-256 hashes
before loading. The models are public; no token is required. A Hub warning may
suggest `HF_TOKEN` for higher rate limits; you can continue without one. Later
calls reuse the Hugging Face cache, normally `~/.cache/huggingface/hub`. Use
`cache_dir` in Python or `--cache-dir` on the CLI to choose its location.
Offline mode (`local_files_only=True` or
`--local-files-only`) requires the files to be cached already; see
[Offline Use](#offline-use) for loading a local bundle. Inference runs locally;
the download requests contain model identifiers, not your input matrix.

For GPU inference, use `device="cuda"` or `device="cuda:1"`. The default precision
is `amp_bfloat16`, matching the published inference recipe. The CPU quick start
explicitly uses `precision="float32"` for compatibility; use this also on GPUs
without BF16 support. Precision and hardware can affect predictions. The device
defaults to CPU even when a GPU is available.

## Prepare Your Data

- Supply a nonempty 2D numerical matrix, with at least one observed value.
  Each cell is a value to keep or impute; exclude row IDs, column labels, and
  free text. For a rating matrix, rows can be users and columns can be items.
- Mark missing entries with `np.nan`. Zeros and sentinel values such as `-999`
  count as observations unless you explicitly replace them with `np.nan`.
- Use a floating-point array to represent `np.nan`. Strings, object arrays,
  complex values, and infinities are rejected. Encode categorical data separately.
- Keep the original row/column orientation. Entirely missing rows or columns
  have limited observational support; an entirely missing matrix is rejected.

The CLI accepts `.npy` arrays, not CSV or Excel files. For a numeric CSV with
one header row, no ID column, and blank or `NaN` missing cells, convert it first:

```python
import numpy as np

matrix = np.genfromtxt("data.csv", delimiter=",", skip_header=1, ndmin=2)
np.save("matrix.npy", matrix)
```

Use `skip_header=0` if the CSV has no header. Keep labels separately; NumPy
arrays and the completed output contain only the numerical values. Inspect
the loaded array before inference, especially if your CSV has mixed types.

## Command Line

Create an example input file by running this Python snippet, or use the CSV
conversion above:

```python
import numpy as np

matrix = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 6.0],
    [3.0, 6.0, 9.0],
])
np.save("matrix.npy", matrix)
```

Then run MatrixFormer on CPU from the directory containing `matrix.npy`:

```bash
python -m matrixformer --input matrix.npy --output completed.npy --device cpu --precision float32
```

Read the result from Python:

```python
import numpy as np

completed = np.load("completed.npy")
print(completed)
```

For a BF16-capable NVIDIA GPU, use `--device cuda` and optionally omit
`--precision float32` to use BF16. Relative input, output,
cache, and checkpoint paths are relative to your current working directory, not
the repository directory.

The command prints a JSON summary, not the completed matrix. `--output` saves
the array and overwrites an existing output file; it must differ from the input
path. Without `--output`, nothing is saved. Without `--input`, the CLI runs the
synthetic demo instead of reading your data. `matrixformer` is an equivalent
console command; `python -m matrixformer` ensures you use the active Python
environment. Run `python -m matrixformer --help` for all options and defaults.

## Offline Use

On a machine with internet access, download the model into a portable directory
(this step does not run inference):

```bash
python -m matrixformer --download-only --local-dir ./checkpoints
```

Copy the entire `checkpoints/` directory to the offline machine. Install Python,
PyTorch, and this package there as well; the checkpoint bundle contains weights
and configs, not the runtime or its dependencies. Then run:

```bash
python -m matrixformer --local-dir ./checkpoints --input matrix.npy --output completed.npy --precision float32
```

Or load that directory from Python, using `matrix` from the earlier example:

```python
from matrixformer import MatrixFormer

model = MatrixFormer.from_pretrained(
    local_dir="./checkpoints", device="cpu", precision="float32",
)
completed = model.impute(matrix)
```

`local_dir` is read-only during inference and makes no network requests. Its
value must be the bundle root (`checkpoints/` here), not a `.pth` file. Its layout is:

```text
checkpoints/
  config.json
  best_validation_checkpoint.pth
```

Old bundles do not match the new checkpoint; download this release before offline use.

Alternatively, download to the default Hub cache with
`python -m matrixformer --download-only`, then use `local_files_only=True` in Python or
`--local-files-only` on the command line. The installed package includes its
own architecture code and decoder-bin constants. No files from another source
checkout are needed, including when running offline.

If you set `cache_dir` or `--cache-dir` while downloading, use the same setting
when loading offline.

## Context and Memory

The default context is at most 2000 rows by 1000 columns. Larger matrices use
overlapping windows. Column normalization is computed from the observed values
of the whole supplied matrix before windowing.

```python
model = MatrixFormer.from_pretrained(
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
memory for the hidden grid and the single network. Start with the small CPU demo;
use a GPU for larger matrices.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named matrixformer` or command not found | Activate the environment used for installation, run `python -m pip install .` from this checkout, then use `python -m matrixformer`. |
| CUDA unavailable or BF16 unsupported | Use `--device cpu --precision float32`, or install a CUDA-enabled PyTorch build compatible with your GPU and driver. On older CUDA GPUs use `--precision float32`. |
| Download fails or offline cache is empty | The first run needs access to Hugging Face. Prepare the bundle in [Offline Use](#offline-use) on a connected machine. A token is not required for these public weights. |
| `Missing artifact` with `--local-dir` | Download the bundle first and pass its root directory. Inference with `--local-dir` never downloads missing files. |
| Out of memory | Try smaller windows, such as `--max-rows 128 --max-cols 64`. Only one network is loaded. Windows can change predictions; weight memory is still required. |
| JSON prints but no completed file appears | Pass `--output completed.npy`; load it with `np.load`. JSON contains diagnostics, not the completed array. |

## Selected Checkpoint and Evaluation

MatrixFormer has 77,998,496 parameters, eight row/column attention blocks, width
768, 16 heads, 12 context tokens and 5,000 output bins. Point predictions use the
distributional mean and the standard matrix orientation.

| Benchmark | MatrixFormer NRMSE |
| --- | ---: |
| MissBench | 1.256252 |
| UCI | 1.285373 |

These results cover 2,091 MissBench and 3,058 UCI cases.
Seeds, rates and patterns are averaged within each dataset before datasets are
weighted equally. The frozen primary metric uses **held-out-truth column SD**.
Separately saved observed-column-SD NRMSE is 4.000035 for MissBench and
1.045013 for UCI; these definitions are not interchangeable. Selection
used earlier UCI/MissBench monitoring, so these are not untouched holdout tests.
RecSys and causal evaluations of these weights were pending at release.

Input normalization uses observed values only. Ground-truth values are used only
for evaluation scoring, never to normalize inference inputs. The generic wrapper
does not implement rating-specific bias centering, user-residual scaling or
clipping. Benchmark masks, preprocessing and context settings must also match to
reproduce benchmark results. No evaluation datasets or runners are included here.

Observed values must be exactly representable as float64; lossy integer or
extended-precision conversions are rejected rather than silently changing them.
Extreme values, unusual missingness patterns, or matrix structure outside the
synthetic training distribution can reduce accuracy. Predictions are means,
not calibrated uncertainty estimates.

## Repository Layout

```text
README.md                  Installation, usage, and limitations
pyproject.toml             Package metadata and three runtime dependencies
src/matrixformer/
  __init__.py              Public API
  __main__.py              CLI and synthetic demo
  imputer.py               Input handling and single-network inference
  artifacts.py             Pinned downloads and checksum verification
  _model.py                Shared transformer architecture
  _inference.py            Streaming and predictive-mean decoding
  training/                Synthetic priors, loss, and standalone trainer
  assets/checkpoints.json  Release hashes and provenance
  assets/borders.pt        Small decoder-bin constants, not a dataset
```

The inference runtime was extracted from the published `tabular` source revision
`2cf54ac47dbcc1154b35e875b527ed0c42a9f416`. The checkpoint keys and inference
arithmetic retain the released architecture. `assets/checkpoints.json` pins Hugging Face release
`4991727bdefa24bb0acd9cd7cfa29d8577b6e8f6` and identifies the selected
final MatrixFormer checkpoint and its minimal config.

The release is checked by strict state-dictionary loading, exact safetensors
conversion and CPU inference with observed entries and input arrays preserved.
These are implementation checks, not benchmark accuracy claims.

## Citation

If you use MatrixFormer, please cite [our paper](https://arxiv.org/abs/2610.06751):

```bibtex
@article{saha2026matrixformer,
  title={MatrixFormer: A Foundation Model for Matrix Completion},
  author={Saha, Dwaipayan and Feitelberg, Jacob and Choi, Kyuseong and Dwivedi, Raaz and Agarwal, Anish},
  journal={arXiv preprint arXiv:2610.06751},
  year={2026}
}
```
