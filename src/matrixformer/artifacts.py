"""Pinned, integrity-checked downloads of the released model artifacts."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from huggingface_hub import hf_hub_download


def manifest():
    resource = files("matrixformer").joinpath("assets/checkpoints.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def verify_file(path, expected_sha256):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}; remove this file and download it again")


def resolve_experts(names, *, local_dir=None, cache_dir=None, local_files_only=False):
    metadata = manifest()
    resolved = {}
    for name in names:
        spec = metadata["experts"][name]
        paths = {}
        for kind in ("config", "checkpoint"):
            if local_dir is not None:
                path = Path(local_dir) / spec[kind]
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Missing artifact: {path}. Run "
                        f"`matrixformer --download-only --local-dir {local_dir}` first."
                    )
            else:
                path = Path(hf_hub_download(
                    metadata["repo_id"], spec[kind], revision=metadata["revision"],
                    cache_dir=cache_dir, local_files_only=local_files_only,
                ))
            verify_file(path, spec[f"{kind}_sha256"])
            paths[kind] = path
        resolved[name] = paths
    return resolved


def download_checkpoints(local_dir=None, *, model="plus", cache_dir=None):
    """Download and verify the frozen artifacts; no authentication is required.

    With local_dir, preserve the Hugging Face matrixformer/{impute,recsys}
    directory layout for subsequent offline loading. Otherwise use the Hub cache.
    """
    expert_names = {"plus": ("impute", "recsys"), "impute": ("impute",), "recsys": ("recsys",)}
    if model not in expert_names:
        raise ValueError("model must be 'plus', 'impute', or 'recsys'")
    names = expert_names[model]
    if local_dir is None:
        return resolve_experts(names, cache_dir=cache_dir)
    metadata = manifest()
    for name in names:
        for kind in ("config", "checkpoint"):
            hf_hub_download(
                metadata["repo_id"], metadata["experts"][name][kind],
                revision=metadata["revision"], local_dir=local_dir,
            )
    return resolve_experts(names, local_dir=local_dir)
