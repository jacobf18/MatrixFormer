"""Reproducible four-domain synthetic matrix prior."""

from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
import json
import math
import random

import numpy as np
import torch

from . import _masks
from ._latent import LatentFactorPrior
from ._neural_masks import mar_config, mar_block_config
from ._ratings import sample_rating_episode
from ._spectral import GeometrySpectrumGenerator
from ._support import normalize_data, normalize_remaining_support, apply_shrunk_support_mask, preserved_rng
from .scientific_panels import TRAIN_FAMILIES, MASK_PATTERNS, sample_scientific_episode
from .scientific_processes import PROCESS_FAMILIES, sample_process_episode, sample_score_panel


DOMAINS = ("science", "imputation", "recsys", "score_panel")
WEIGHTS = dict(science=0.35, imputation=0.35, recsys=0.20, score_panel=0.10)
SCIENCE_FAMILIES = TRAIN_FAMILIES + PROCESS_FAMILIES
PATTERNS = dict(_masks.ProfileMixedPattern.pattern_classes)
PATTERNS.update(
    mar_neural=_masks.MARNeuralNetwork,
    mar_block_neural=_masks.MARBlockNeuralNetwork,
    mnar_latent_factor=_masks.MNARLatentFactorPattern,
    mnar_polarization=_masks.MNARPolarizationPattern,
    mnar_soft_polarization=_masks.MNARSoftPolarizationPattern,
)
COMPONENTS = ("low_rank_mcar", "slow_tail_mcar", "steep_tail_mcar", "anchor_latent_mcar")
NEURAL_PATTERNS = ("mar_neural", "mar_block_neural")
SMALL_SHAPES = ((32, 8), (64, 16), (96, 24), (128, 32), (192, 48), (256, 64))
MEDIUM_SHAPES = ((384, 96), (512, 128), (768, 128))


@dataclass
class Episode:
    source: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    metadata: dict

    def to(self, device):
        return Episode(self.source.to(device), self.target.to(device), self.mask.to(device), self.metadata)


def specification(domain, ordinal, seed):
    random_shape = np.random.default_rng(seed)
    bucket = int(random_shape.integers(1000))
    if bucket < 800:
        shape = SMALL_SHAPES[int(random_shape.integers(len(SMALL_SHAPES)))]
    elif bucket < 980:
        shape = MEDIUM_SHAPES[int(random_shape.integers(len(MEDIUM_SHAPES)))]
    else:
        shape = (1024, 256) if bucket < 998 else ((1536, 512) if bucket == 998 else (2000, 1000))
    random_spec = np.random.default_rng(seed + 1)
    result = dict(domain=domain, seed=seed, shape=list(shape))
    if domain == "science":
        result.update(family=SCIENCE_FAMILIES[ordinal % len(SCIENCE_FAMILIES)],
                      pattern=MASK_PATTERNS[(ordinal // len(SCIENCE_FAMILIES)) % len(MASK_PATTERNS)])
    elif domain == "imputation":
        if ordinal % 2 == 0:
            result.update(kind="native", component=(ordinal // 2) % 8,
                          regime="area_scaled" if random_spec.random() < 0.5 else "legacy")
        else:
            pair = (ordinal // 2) % (len(PATTERNS) * len(COMPONENTS))
            family = tuple(PATTERNS)[pair // len(COMPONENTS)]
            shapes = ((24, 12), (48, 24)) if family in NEURAL_PATTERNS else ((240, 80), (1024, 256))
            result.update(kind="structured", family=family, component=COMPONENTS[pair % len(COMPONENTS)],
                          shape=list(shapes[int(random_spec.random() >= 0.85)]),
                          regime="area_scaled" if random_spec.random() < 0.5 else "legacy",
                          requested_missingness=0.2 if random_spec.random() < 0.5 else 0.6)
    elif domain == "recsys":
        result["shape"] = [max(128, shape[0]), max(48, shape[1])]
    elif domain == "score_panel":
        result["shape"] = [max(96, min(512, shape[0])), max(16, min(128, shape[1]))]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return result


def training_specs(step, seed=1810000000):
    if step < 1 or seed < 0 or seed + 64 * step >= 2**32:
        raise ValueError("Require a positive step and seeds within the uint32 range")
    return [specification(domain, step - 1, seed + 16 * (4 * (step - 1) + index))
            for index, domain in enumerate(DOMAINS)]


def validation_specs(seed=2310000000):
    specifications = []
    for family in SCIENCE_FAMILIES:
        for pattern in ("bursts", "self_mnar", "covariate_mar"):
            specifications.append(dict(domain="science", family=family, pattern=pattern, shape=[192, 32]))
    for component in range(8):
        for regime in ("legacy", "area_scaled"):
            specifications.append(dict(domain="imputation", kind="native", component=component,
                                       regime=regime, shape=[256, 64]))
    for family in PATTERNS:
        for component in ("slow_tail_mcar", "anchor_latent_mcar"):
            specifications.append(dict(domain="imputation", kind="structured", family=family,
                                       component=component, regime="area_scaled", requested_missingness=0.6,
                                       shape=[48, 24] if family in NEURAL_PATTERNS else [240, 80]))
    for domain in ("recsys", "score_panel"):
        for shape in ((128, 48), (256, 96), (512, 128)):
            for repeat in range(2):
                specifications.append(dict(domain=domain, shape=list(shape)))
    specifications.extend([
        dict(domain="science", family="smooth_local", pattern="bursts", shape=[1024, 256]),
        dict(domain="imputation", kind="native", component=2, regime="area_scaled", shape=[1024, 256]),
    ])
    return [dict(spec, seed=seed + 16 * index) for index, spec in enumerate(specifications)]


class ScaledSpectrum(GeometrySpectrumGenerator):
    def __init__(self, config, gain):
        super().__init__(config)
        self.gain = gain

    def sample_spectrum(self, rank):
        values, parameters = super().sample_spectrum(rank)
        return values * self.gain, parameters


def latent_matrix(config):
    """Keep the optional-transform RNG draws even with warping and quantization disabled."""
    rows = int(np.random.randint(config["num_rows_low"], config["num_rows_high"] + 1))
    columns = int(np.random.randint(config["num_cols_low"], config["num_cols_high"] + 1))
    rank = int(np.random.randint(config["latent_rank_low"], min(config["latent_rank_high"], rows, columns) + 1))
    generator = LatentFactorPrior(config)
    values = generator._sample_latent_vectors(rows, rank) @ generator._sample_latent_vectors(columns, rank).T
    np.random.rand()
    np.random.rand()
    return torch.as_tensor(values, dtype=torch.float32)


class SyntheticPrior:
    """Draw one matrix per domain, with no real-data inputs or model targets."""

    def __init__(self, *, max_rows=None, max_cols=None, smoke=False):
        for value in (max_rows, max_cols):
            if value is not None and value < 16:
                raise ValueError("Context caps must be at least 16")
        self.components = json.loads(files(__package__).joinpath("components.json").read_text())
        self.max_rows, self.max_cols, self.smoke = max_rows, max_cols, smoke

    def _configuration(self, component, shape, *, native):
        entry = self.components[component] if isinstance(component, int) else next(
            entry for entry in self.components if entry["name"] == component)
        config = deepcopy(entry["config"])
        if not entry["name"].startswith("anchor"):
            config.update(num_rows_low=shape[0], num_rows_high=shape[0],
                          num_cols_low=shape[1], num_cols_high=shape[1])
        if native:
            for axis in ("rows", "cols"):
                size = int(np.random.randint(config[f"num_{axis}_low"], config[f"num_{axis}_high"] + 1))
                config[f"num_{axis}_low"] = config[f"num_{axis}_high"] = size
        return entry["name"], config

    def _imputation(self, spec):
        native = spec["kind"] == "native"
        with preserved_rng(spec["seed"]):
            if native:
                random.random()
            name, config = self._configuration(spec["component"], spec["shape"], native=native)
            if name.startswith("anchor"):
                source = latent_matrix(config)
                target = source.clone()
            else:
                gain = max(1.0, math.sqrt(spec["shape"][0] * spec["shape"][1] / 19200))
                bundle = ScaledSpectrum(config, gain if spec["regime"] == "area_scaled" else 1.0).generate_matrix_bundle()
                source, target = bundle["full_matrix"], bundle["target_matrix"]
            if native:
                source, (center, scale) = normalize_data(source, return_scaling=True)
                target = normalize_data(target, mean=center, std=scale)
                constructor = _masks.MCARPattern if name.endswith("_mcar") else _masks.ProfileMixedPattern
                source = constructor(config)._induce_missingness(source.clone())
                source, (center, scale) = normalize_data(source, return_scaling=True)
                target = normalize_data(target, mean=center, std=scale)
                transposed = random.random() < 0.5
                if transposed:
                    source, target = source.T.contiguous(), target.T.contiguous()
        if native:
            source, target, query, metadata = apply_shrunk_support_mask(
                source[None], target[None], "broad_support", seed=spec["seed"] + 10000000, variance_df=1.0)
            return Episode(source, target, query, dict(metadata, component=name, transposed=transposed))
        mechanism_source = normalize_data(source)
        mask_config = _masks.ProfileMixedPattern(config)._build_pattern_config(spec["requested_missingness"])
        mask_config.update(mar_config=deepcopy(mar_config), mar_block_config=deepcopy(mar_block_config))
        with preserved_rng(spec["seed"] + 1):
            query = torch.isnan(PATTERNS[spec["family"]](mask_config)._induce_missingness(mechanism_source.clone()))
        source, target, _, _ = normalize_remaining_support(source[None], target[None], (~query)[None], variance_df=1.0)
        return Episode(source, target, query[None], dict(component=name, family=spec["family"]))

    def _draw(self, spec, device):
        domain, shape = spec["domain"], spec["shape"]
        if domain in ("science", "score_panel"):
            if domain == "score_panel":
                generated = sample_score_panel(spec["seed"], rows=shape[0], columns=shape[1])
            else:
                generator = sample_process_episode if spec["family"] in PROCESS_FAMILIES else sample_scientific_episode
                generated = generator(spec["seed"], family=spec["family"], pattern=spec["pattern"],
                                      rows=shape[0], columns=shape[1])
            return Episode(torch.from_numpy(generated.source)[None], torch.from_numpy(generated.target)[None],
                           torch.from_numpy(generated.query)[None], generated.metadata)
        if domain == "imputation":
            return self._imputation(spec)
        if domain != "recsys":
            raise ValueError(f"Unknown domain: {domain}")
        generated = sample_rating_episode(
            spec["seed"], rows=shape[0], cols=shape[1], device=device, query_limit=4096,
            density_range=(0.004, 0.08), parent_rows=128 if self.smoke else 12288,
            parent_cols=64 if self.smoke else 4096, normalization="two_way_user_scale",
            point_target="expected_rating", item_popularity_sigma_range=(1.4, 2.8),
            context_ordering="first_observation", query_sampling="user_balanced", preference_family="mixed",
            exclude_unknown_queries=True, uniform_noise_range=(0.0, 0.15), user_noise_log_std_range=(0.15, 0.6),
        )
        return Episode(generated.source, generated.target, generated.rating_query, generated.metadata)

    def draw(self, spec, *, device="cpu"):
        spec = deepcopy(spec)
        requested_shape = list(spec["shape"])
        if self.smoke:
            spec["shape"] = [32, 16]
        else:
            spec["shape"] = [min(requested_shape[0], self.max_rows or requested_shape[0]),
                             min(requested_shape[1], self.max_cols or requested_shape[1])]
        requested_seed = spec["seed"]
        for retry in range(16):
            spec["seed"] = requested_seed + retry
            episode = self._draw(spec, device)
            if not episode.mask.any() or not torch.isfinite(episode.source).any():
                continue
            if not torch.isfinite(episode.target).all() or torch.isinf(episode.source).any():
                raise ValueError(f"Nonfinite synthetic data: {spec}")
            if (episode.mask & torch.isfinite(episode.source)).any():
                raise ValueError("A training query must be hidden from the model")
            episode.metadata = dict(episode.metadata, domain=spec["domain"], specification=spec,
                                    requested_seed=requested_seed, rejected_draws=retry, requested_shape=requested_shape,
                                    rows=episode.source.shape[1], columns=episode.source.shape[2],
                                    observed_cells=int(torch.isfinite(episode.source).sum()), query_cells=int(episode.mask.sum()))
            return episode.to(device)
        raise ValueError(f"No usable synthetic query after 16 seeds: {spec}")
