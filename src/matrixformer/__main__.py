"""Command-line inference and checkpoint downloads."""

import argparse
import json
from pathlib import Path

import numpy as np

from . import (
    MatrixFormer, MatrixFormerImpute, MatrixFormerPlus, MatrixFormerRecSys, download_checkpoints,
)
from .imputer import prepare_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen MatrixFormer matrix completion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=("matrixformer", "impute", "recsys", "ensemble", "plus"),
        default="matrixformer",
        help="MatrixFormer (matrixformer), a single specialist (impute or recsys), or the "
        "MatrixFormer* ensemble of both specialists (ensemble; plus is the former name)",
    )
    parser.add_argument(
        "--input", type=Path,
        help="2D numeric .npy array with NaN for missing entries; omitted runs a synthetic demo",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Save completed .npy array, overwriting existing output; omitted prints only a summary",
    )
    parser.add_argument("--device", default="cpu", help="Inference device: cpu, cuda, or cuda:N")
    parser.add_argument(
        "--precision", choices=("amp_bfloat16", "float32"), default="amp_bfloat16",
        help="Inference precision; use float32 on hardware without BF16 support",
    )
    parser.add_argument("--max-rows", type=int, default=2000, help="Context rows; 0 uses the full axis")
    parser.add_argument("--max-cols", type=int, default=1000, help="Context columns; 0 uses the full axis")
    parser.add_argument(
        "--local-dir", type=Path,
        help="Existing offline bundle root; with --download-only, the download destination",
    )
    parser.add_argument("--cache-dir", type=Path, help="Override the Hugging Face cache directory")
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="Read from the Hub cache without network access",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Download weights without running inference",
    )
    args = parser.parse_args()
    if args.max_rows < 0 or args.max_cols < 0:
        parser.error("Context limits must be nonnegative")
    if args.download_only and (
        args.input is not None or args.output is not None or args.local_files_only
    ):
        parser.error("--download-only cannot be combined with input/output or --local-files-only")
    for name in ("input", "output"):
        path = getattr(args, name)
        if path is not None and path.suffix != ".npy":
            parser.error(f"--{name} must have a .npy suffix")
    if args.input is not None and args.output is not None:
        same_path = args.input.resolve() == args.output.resolve()
        same_file = (
            args.input.exists() and args.output.exists() and args.input.samefile(args.output)
        )
        if same_path or same_file:
            parser.error("--output must differ from --input, including links to the same file")
    return args


def main():
    args = parse_args()
    if args.download_only:
        paths = download_checkpoints(args.local_dir, model=args.model, cache_dir=args.cache_dir)
        print(json.dumps(paths, default=str, indent=2))
        return
    if args.input is None:
        generator = np.random.default_rng(42)
        matrix = generator.normal(size=(12, 3)) @ generator.normal(size=(3, 9))
        matrix.flat[generator.choice(matrix.size, size=30, replace=False)] = np.nan
    else:
        matrix = np.load(args.input, allow_pickle=False)
    matrix = prepare_matrix(matrix)
    factory = {
        "matrixformer": MatrixFormer, "impute": MatrixFormerImpute, "recsys": MatrixFormerRecSys,
        "ensemble": MatrixFormerPlus, "plus": MatrixFormerPlus,
    }[args.model]
    model = factory.from_pretrained(
        device=args.device, precision=args.precision, max_rows=args.max_rows or None,
        max_cols=args.max_cols or None, local_dir=args.local_dir, cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    completed = model.impute(matrix)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, completed, allow_pickle=False)
    observed = np.isfinite(matrix)
    report = {
        "model": args.model, "device": args.device, "precision": args.precision,
        "shape": list(matrix.shape), "missing_entries": int(np.isnan(matrix).sum()),
        "all_predictions_finite": bool(np.isfinite(completed).all()),
        "observed_entries_preserved": bool(np.array_equal(completed[observed], matrix[observed])),
    }
    if args.model in ("ensemble", "plus"):
        report["blend_decision"] = model.last_decision
    if args.output is not None:
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
