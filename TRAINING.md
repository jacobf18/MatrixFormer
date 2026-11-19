# Synthetic training

Train one MatrixFormer from random initialization using synthetic matrices.
No real datasets, pretrained weights, external research checkout, or model
targets are needed. The inference API and its dependencies remain separate.

## Run

From this repository:

```bash
python -m pip install -e ".[training]"
python -m matrixformer.training --output runs/matrixformer --device cuda
```

`matrixformer-train` is an equivalent entry point. The default architecture is
the 77,998,496-parameter MatrixFormer: eight axial blocks, width 768, 16 heads,
12 context tokens per axis, and the packaged 5,000-bin full-support output
distribution. CUDA uses BF16 autocast with FP32 parameters and loss calculations;
CPU defaults to FP32.

The default run is **250,000 optimizer updates and 1,000,000 training matrices**.
Each update accumulates gradients from four matrices, one per domain, before
one AdamW step. Validation matrices and rejected degenerate draws do not count
toward that total. The learning rate defaults to `1e-4`, with 1,000 warmup updates
and cosine decay to 10% of the peak; gradient norm is clipped at 1.

This is a runnable training recipe, not a guarantee of a particular benchmark
score. Tune optimization using synthetic validation rather than real test sets.

### Small installation check

```bash
python -m matrixformer.training --output runs/smoke --smoke --device cpu --threads 2
```

This runs two updates (eight training matrices) with a 362,318-parameter model,
small contexts, a smaller rating population, and four fixed validation matrices.
It checks training, validation, and checkpoint writing, not model accuracy.
Use a fresh output directory for each new run.

### Resume and checkpoints

Repeat the same recipe options with `--resume`:

```bash
python -m matrixformer.training --output runs/matrixformer --device cuda --resume
```

Resume restores the model, optimizer, RNG, and completed-update count. Data
seeds depend on update and domain, so the next update draws the same matrices
as an uninterrupted run. Recipe options, including the total `--steps` horizon,
must match. CUDA kernels can still introduce numerical nondeterminism.
`--stop-after 100` saves and exits at update 100 without changing the
learning-rate horizon; omit it when resuming to continue to the original target.

Each output directory contains:

| File | Contents |
| --- | --- |
| `config.json` | Architecture for inference |
| `training.json` | Recorded recipe and hyperparameters |
| `history.jsonl` | Losses, timing, gradient norms, sampled specs, and validation metrics |
| `last.pth` | Atomic resume checkpoint, including optimizer state |
| `best_validation_checkpoint.pth` | Model selected on fixed synthetic validation |

Checkpoints default to every 1,000 updates and at normal exit. Validation runs
every 1,000 updates and at the requested final training step; a deliberately
early `--stop-after` does not force validation. An interruption can require
replaying updates since the last checkpoint. `--save-every`, `--validate-every`,
and `--log-every` control these intervals. Allow several GB for full-model
checkpoints, optimizer state, and atomic temporary files.

Load your own saved model without changing the pinned published model:

```python
from matrixformer import MatrixFormer

model = MatrixFormer.from_checkpoint(
    "runs/matrixformer/best_validation_checkpoint.pth",
    device="cuda",
)
completed = model.impute(matrix)
```

`from_checkpoint` uses `config.json` beside the checkpoint and strict parameter
loading. It does not claim that locally trained weights match the published
release hash. `from_pretrained()` continues to verify the published release.

## Prior

| Domain | Matrices per update | Loss weight | Structure |
| --- | ---: | ---: | --- |
| Science | 1 | 0.35 | Smooth sensors, dynamic factors, transport, nonlinear calibration, grouped regimes, chemical mixtures, treatment processes, assays, local smoothness, and spatial profiles |
| Imputation | 1 | 0.35 | Low-rank, slow/steep spectral decay, and heterogeneous latent factors |
| Ratings | 1 | 0.20 | Ordinal user–item ratings, Gaussian/clustered preferences, and nonuniform exposure |
| Score panels | 1 | 0.10 | Bounded latent scores and partially observed new rows |

Thus matrix counts are 25% per domain; **35/35/20/10 are loss weights**, not draw
probabilities. All four draws are processed sequentially to limit memory.

- Science cycles ten families and six masks: isolated gaps, bursts, network
  outages, scheduled assays, self-MNAR, and covariate-MAR. Some episodes include
  observed time features. These are domain-inspired statistical generators,
  not calibrated physical simulators.
- Imputation alternates native and structured-mask episodes. Native draws cycle
  eight combinations of three spectral families or latent factors with MCAR
  or mixed missingness, then optionally transpose and thin row/column support.
  Structured draws cross four value generators with twelve mask mechanisms.
  The mask implementations live in `_masks.py`; their implementation names
  are retained as identifiers, not a new classification of MCAR/MAR/MNAR.
- Spectral generators mix Haar, coherent, smooth, and offset singular-vector
  geometry. Half of imputation episodes request singular-value amplitude
  scaling by `max(1, sqrt(rows * columns / 19200))`. Spectral targets are clean
  signals; observed inputs include sampled noise. Latent-factor matrices keep
  intrinsic sizes of 10–50 rows and 5–50 columns.
- Ratings are generated within a 12,288-user by 4,096-item population, then
  cropped in first-observation order. Density is log-uniform from 0.004 to 0.08.
  Observed-support bias normalization and shrinkage of user scales precede
  cropping; rating queries are user-balanced and exclude unknown users/items.
  The loss target is the normalized held-out realized rating.
- Source, target, and query mask are separate. Queries must be hidden from the
  input. Empty-query draws are retried deterministically, at most 16 times,
  with the retry count recorded. Nonfinite data raises an error.

Synthetic native imputation construction includes complete-matrix
standardization before masking, followed by observed-support normalization.
Structured imputation uses observed-only normalization with within-column
variance shrinkage. Science and score panels use observed-column means and
standard deviations. Real-data inference always estimates input normalization
from observed values only; it does not have access to hidden targets.

## Objective and validation

For each queried cell, the loss is

`NLL_weight * NLL + Huber_weight * Huber(predictive_mean, target) + 0.05 * CDF_error`.

The density is piecewise uniform inside the output bins with half-normal tails.
An interior target between borders `i` and `i + 1` is assigned to output bin `i`.
NLL weight is 0.2 for ratings and 1 elsewhere. Huber uses delta 1 and weights
4 for science, 2 for imputation, and 1 for ratings/score panels. The CDF term
compares cumulative bin probabilities with the target step function at bin
midpoints. Losses average over queried cells within each matrix, then use the
domain weights above. There is no loss on observed input entries.

The default fixed validation suite has 84 independently seeded synthetic
matrices. Each matrix contributes its RMSE in its generated normalized units;
these are averaged within domains and combined with the same domain weights.
This score selects `best_validation_checkpoint.pth`; it is not MissBench/UCI
NRMSE. Validation never participates in backpropagation. Training and validation
seed ranges must not overlap.

## Memory and size controls

Base size sampling is 80% from 32×8 through 256×64, 18% from 384×96 through
768×128, 1.8% at 1024×256, and 0.1% each at 1536×512 and 2000×1000.
Structured masks, latent anchors, ratings, and score panels have domain-specific
shape overrides in `prior.py`; this base schedule is not the final shape
histogram. Optional science time features add three columns.

The full schedule needs substantial GPU memory. To start with bounded contexts:

```bash
python -m matrixformer.training --output runs/smaller \
  --device cuda --max-rows 256 --max-cols 64
```

These caps change requested generated contexts before native transposition;
latent anchors retain intrinsic sizes, science can append time features, and
the rating parent population remains unchanged. They are recorded in the
recipe. They do not crop attention sequences within a forward pass.

`--attention-batch-size`, `--cell-batch-size`, `--encoder-row-batch-size`, and
`--decoder-cell-batch-size` bound intermediate work while retaining full context.
Activation checkpointing recomputes intermediates during backward; use
`--no-checkpointing` only when memory permits. Hidden grids and their gradients
still consume memory proportional to matrix size. OOM failures are not silently
skipped or replaced with easier matrices. Use `--help` for all options.
