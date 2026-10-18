# MatrixFormer

Fill missing entries in numerical matrices with pretrained transformers.
MatrixFormer+ combines a general-purpose model and a recommender-system model,
choosing their blend from the observed entries in your matrix. Supply a 2D NumPy
array with `np.nan` for missing entries. The output preserves observed entries
exactly. You do not need to train or fine-tune either model.

This is a standalone, inference-only repository. All model, streaming,
normalization, and blending code lives here. It does not import `tabimpute`,
depend on the `tabular` repository, or contain datasets, training jobs, or
benchmark infrastructure.

- [Project page](https://jacobf18.github.io/MatrixFormer/)
- [Hugging Face weights and configs](https://huggingface.co/Tabimpute/TabImpute)
- [Paper](https://arxiv.org/abs/2610.06751)
- [Paper source](https://github.com/jacobf18/MatrixFormerPaper)

## Install and First Run

You need Git, Python 3.10 or newer, and internet access for installation and the
first model download. A GPU and a Hugging Face account are optional. Allow disk
space for PyTorch and its dependencies as well as about 624 MB for both models.

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

The last command downloads the models and fills 30 missing entries in a small
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

## Choose a Model

| Model | Python class | CLI option | Weights |
| --- | --- | --- | --- |
| MatrixFormer+ | `MatrixFormerPlus` | `--model plus` (default) | Both experts, about 624 MB |
| MatrixFormer-Impute | `MatrixFormerImpute` | `--model impute` | General expert only, about 312 MB |
| MatrixFormer-RecSys | `MatrixFormerRecSys` | `--model recsys` | Recommender expert only, about 312 MB |

Start with MatrixFormer+ for the two-model blend, or choose MatrixFormer-Impute
to run just the general model with lower weight memory requirements.
MatrixFormer-RecSys runs just the recommender specialist, which was fine-tuned
on sparse synthetic rating matrices. The
Hugging Face repository is named `Tabimpute/TabImpute` for historical reasons;
the MatrixFormer files are under its `matrixformer/` directory. There is no
separate MatrixFormer+ checkpoint.

## Python

```python
import numpy as np
from matrixformer import MatrixFormerPlus

matrix = np.array([
    [1.0, 2.0, np.nan],
    [2.0, np.nan, 6.0],
    [3.0, 6.0, 9.0],
])

model = MatrixFormerPlus.from_pretrained(device="cpu", precision="float32")
completed = model.impute(matrix)

print(completed)
print(model.last_decision)
```

`completed` is a float64 NumPy array with the same shape and units as `matrix`.
The input is left unchanged. Column normalization and conversion back to the
original units happen internally. Reuse the model object for multiple matrices.

When `local_dir` is omitted, `MatrixFormerPlus.from_pretrained()` automatically
downloads both experts (about 624 MB of weights) and their matching configs from
[Tabimpute/TabImpute on Hugging Face](https://huggingface.co/Tabimpute/TabImpute).
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

The single general expert is also available:

```python
from matrixformer import MatrixFormerImpute

model = MatrixFormerImpute.from_pretrained(device="cpu", precision="float32")
completed = model.impute(matrix)
```

Loading MatrixFormer-Impute downloads only its own checkpoint, about 312 MB.

The recommender specialist can be run on its own the same way:

```python
from matrixformer import MatrixFormerRecSys

model = MatrixFormerRecSys.from_pretrained(device="cpu", precision="float32")
completed = model.impute(matrix)
```

`MatrixFormerRecSys` downloads only the recommender checkpoint, also about
312 MB. It uses the same generic column normalization as the other classes,
not the paper's rating recipe of user/item bias centering and shrunk user
scales, so its rating predictions differ from the paper's recommender results.

### Understanding the Blend Output

After calling `impute`, `model.last_decision` describes the MatrixFormer+ blend:

- `specialist_weight`: the recommender model's weight, from 0 (general only) to
  1 (specialist only). Intermediate values blend both predictions.
- `general_mse` and `specialist_mse`: column-standardized reconstruction errors
  on visible entries, not test errors or errors in the original units.
- `evidence_cells`: the number of observed entries used to choose the weight.
- `selected`: the expert with the lower individual reconstruction error. This
  label does not replace the blend; `specialist_weight` determines the output.

`MatrixFormerImpute` and `MatrixFormerRecSys` each use one model and have no
`last_decision` attribute.

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

Then run MatrixFormer+ on CPU from the directory containing `matrix.npy`:

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
`--precision float32` to use BF16. Add `--model impute` to use the single general
expert, or `--model recsys` for the single recommender expert. Relative input, output, cache, and checkpoint paths are relative to your
current working directory, not the repository directory.

The command prints a JSON summary, not the completed matrix. `--output` saves
the array and overwrites an existing output file; it must differ from the input
path. Without `--output`, nothing is saved. Without `--input`, the CLI runs the
synthetic demo instead of reading your data. `matrixformer` is an equivalent
console command; `python -m matrixformer` ensures you use the active Python
environment. Run `python -m matrixformer --help` for all options and defaults.

## Offline Use

On a machine with internet access, download both experts into a portable
directory (this step does not run inference):

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
from matrixformer import MatrixFormerPlus

model = MatrixFormerPlus.from_pretrained(
    local_dir="./checkpoints", device="cpu", precision="float32",
)
completed = model.impute(matrix)
```

`local_dir` is read-only during inference and makes no network requests. Its
value must be the bundle root (`checkpoints/` here), not a `.pth` file or the
inner `matrixformer/` directory. Its layout is:

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
`python -m matrixformer --download-only`, then use `local_files_only=True` in Python or
`--local-files-only` on the command line. The installed package includes its
own architecture code and decoder-bin constants. No files from another source
checkout are needed, including when running offline.

If you set `cache_dir` or `--cache-dir` while downloading, use the same setting
when loading offline. Use `--model impute` (or `--model recsys`) for both
download and inference to prepare only that expert. A single-expert bundle
cannot run MatrixFormer+.

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

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `No module named matrixformer` or command not found | Activate the environment used for installation, run `python -m pip install .` from this checkout, then use `python -m matrixformer`. |
| CUDA unavailable or BF16 unsupported | Use `--device cpu --precision float32`, or install a CUDA-enabled PyTorch build compatible with your GPU and driver. On older CUDA GPUs use `--precision float32`. |
| Download fails or offline cache is empty | The first run needs access to Hugging Face. Prepare the bundle in [Offline Use](#offline-use) on a connected machine. A token is not required for these public weights. |
| `Missing artifact` with `--local-dir` | Download the bundle first and pass its root directory. Inference with `--local-dir` never downloads missing files. |
| Out of memory | Try smaller windows, such as `--max-rows 128 --max-cols 64`, or the single expert with `--model impute`. Windows can change predictions; weight memory is still required. |
| JSON prints but no completed file appears | Pass `--output completed.npy`; load it with `np.load`. JSON contains diagnostics, not the completed array. |

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

## Citation

If you use MatrixFormer, please cite [our paper](https://arxiv.org/abs/2610.06751):

```bibtex
@misc{saha2026matrixformerfoundationmodelmatrix,
  title={MatrixFormer: A Foundation Model for Matrix Completion},
  author={Dwaipayan Saha and Jacob Feitelberg and Kyuseong Choi and Raaz Dwivedi and Anish Agarwal},
  year={2026},
  eprint={2610.06751},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.06751}
}
```
