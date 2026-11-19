"""Train one MatrixFormer on synthetic matrices: python -m matrixformer.training."""

import argparse
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .._model import FrozenModel
from .forward import masked_loss
from .objective import DistributionLoss
from .prior import DOMAINS, WEIGHTS, SyntheticPrior, training_specs, validation_specs


MODEL_CONFIG = dict(
    embedding_size=768, mlp_hidden_size=1536, num_attention_heads=16, num_layers=8,
    num_cls=12, num_outputs=5000, rms_norm_eps=1e-6, rope_base=10000.0,
    rope_fraction=0.5, residual_scale_init=1.0, use_absolute_positional_embeddings=False,
    use_rope=True, positional_damping_factor=0.1, attention_dropout=0.0,
    embedding_dropout=0.0, ffn_dropout=0.0, drop_path_rate=0.0,
)


def atomic_save(payload, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def learning_rate(step, config):
    warmup = min(config["warmup_steps"], config["steps"])
    if step <= warmup:
        return config["learning_rate"] * step / max(1, warmup)
    progress = (step - warmup) / max(1, config["steps"] - warmup)
    return config["learning_rate"] * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def precision_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "amp_bfloat16")


@torch.no_grad()
def validate(model, distribution, sampler, specs, device, precision, options):
    model.eval()
    scores = {domain: [] for domain in DOMAINS}

    def squared_error(logits, truth):
        return (distribution.mean(logits) - truth).square()

    try:
        for spec in specs:
            episode = sampler.draw(spec, device=device)
            with precision_context(device, precision):
                mse = masked_loss(model, episode.source, episode.target, episode.mask, squared_error, **options)
            if not torch.isfinite(mse):
                raise FloatingPointError(f"Nonfinite validation error: {spec}")
            scores[spec["domain"]].append(float(mse.sqrt()))
    finally:
        model.train()
    means = {domain: sum(values) / len(values) for domain, values in scores.items()}
    return dict(rmse=means, score=sum(WEIGHTS[domain] * means[domain] for domain in DOMAINS),
                matrices=len(specs))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("float32", "amp_bfloat16"))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--data-seed", type=int, default=1810000000)
    parser.add_argument("--validation-seed", type=int, default=2310000000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-cols", type=int)
    parser.add_argument("--attention-batch-size", type=int, default=32)
    parser.add_argument("--cell-batch-size", type=int, default=4096)
    parser.add_argument("--encoder-row-batch-size", type=int, default=32)
    parser.add_argument("--decoder-cell-batch-size", type=int, default=256)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume OUTPUT/last.pth; use the same recipe options")
    parser.add_argument("--stop-after", type=int, help="Save and exit after this absolute update without changing the LR horizon")
    parser.add_argument("--smoke", action="store_true", help="Tiny model, contexts and rating population; a wiring check, not an accuracy run")
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (2 if args.smoke else 250000)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("Only CPU and CUDA are supported")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; pass --device cpu")
        torch.cuda.set_device(device)
    args.precision = args.precision or ("amp_bfloat16" if device.type == "cuda" else "float32")
    if device.type == "cuda" and args.precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("This GPU does not support BF16; use --precision float32")
    positive = ("steps", "validate_every", "save_every", "log_every", "threads", "attention_batch_size",
                "cell_batch_size", "encoder_row_batch_size", "decoder_cell_batch_size")
    if any(getattr(args, name) < 1 for name in positive) or args.warmup_steps < 0:
        parser.error("Counts must be positive and warmup nonnegative")
    for name in ("learning_rate", "clip_norm", "weight_decay"):
        value = getattr(args, name)
        if not math.isfinite(value) or (value < 0 if name == "weight_decay" else value <= 0):
            parser.error(f"Invalid {name}")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        parser.error("--stop-after must lie between 1 and --steps")
    training_specs(args.steps, args.data_seed)
    validation_end = args.validation_seed + 16 * len(validation_specs())
    if args.validation_seed < 0 or validation_end >= 2**32:
        parser.error("Validation seeds must fit uint32")
    if max(args.data_seed, args.validation_seed) < min(args.data_seed + 64 * args.steps, validation_end):
        parser.error("Training and validation seed ranges must not overlap")
    return args, device


def main():
    args, device = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model_config = dict(MODEL_CONFIG)
    if args.smoke:
        model_config.update(embedding_size=32, mlp_hidden_size=64, num_attention_heads=2, num_layers=2, num_cls=2)
    options = {name: getattr(args, name) for name in ("attention_batch_size", "cell_batch_size",
                                                     "encoder_row_batch_size", "decoder_cell_batch_size")}
    options["use_checkpoint"] = not args.no_checkpointing
    config = {name: value for name, value in vars(args).items() if name not in ("output", "resume", "stop_after")}
    config.update(model=model_config, recipe_version=1, matrices_per_update=4, domain_weights=WEIGHTS)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = args.output.resolve()
    sampler = SyntheticPrior(max_rows=args.max_rows, max_cols=args.max_cols, smoke=args.smoke)
    if args.resume:
        if not (output / "last.pth").is_file():
            raise FileNotFoundError(f"No resumable checkpoint at {output / 'last.pth'}")
        if json.loads((output / "training.json").read_text()) != config:
            raise ValueError("Resume requires identical recipe options, including the total step count")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Choose an empty output directory, or use --resume")
        output.mkdir(parents=True, exist_ok=True)
        (output / "training.json").write_text(json.dumps(config, indent=2) + "\n")
        (output / "config.json").write_text(json.dumps(dict(model=model_config), indent=2) + "\n")
    model = FrozenModel(model_config).to(device)
    distribution = DistributionLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step, best_score = 0, math.inf
    if args.resume:
        payload = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        if payload["config_sha256"] != config_hash:
            raise ValueError("Checkpoint recipe hash mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_step, best_score = payload["step"], payload["best_score"]
        torch.set_rng_state(payload["rng_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(payload["rng_device"], device)
        del payload
    specs = validation_specs(args.validation_seed)
    if args.smoke:
        specs = [next(spec for spec in specs if spec["domain"] == domain) for domain in DOMAINS]
    final_step = args.stop_after or args.steps
    if start_step >= final_step:
        print(json.dumps(dict(status="already_complete", step=start_step, matrices=4 * start_step)))
        return
    model.train()
    print(json.dumps(dict(status="training", device=str(device), output=str(output), parameters=sum(
        parameter.numel() for parameter in model.parameters()), steps=args.steps, smoke=args.smoke)), flush=True)
    for step in range(start_step + 1, final_step + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        rate = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = rate
        losses, draws = {}, []
        for spec in training_specs(step, args.data_seed):
            domain = spec["domain"]
            episode = sampler.draw(spec, device=device)
            with precision_context(device, args.precision):
                loss = masked_loss(model, episode.source, episode.target, episode.mask,
                                   partial(distribution, domain=domain), **options)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}: {episode.metadata}")
            (WEIGHTS[domain] * loss).backward()
            losses[domain] = float(loss.detach())
            draws.append(episode.metadata)
            del episode, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm, error_if_nonfinite=True)
        optimizer.step()
        event = dict(step=step, matrices=4 * step, loss=losses, learning_rate=rate,
                     gradient_norm=float(gradient_norm), seconds=time.monotonic() - started)
        if step % args.validate_every == 0 or step == args.steps:
            metrics = validate(model, distribution, sampler, specs, device, args.precision, options)
            event["validation"] = metrics
            if metrics["score"] < best_score:
                best_score = metrics["score"]
                atomic_save(dict(model=model.state_dict(), step=step, matrices=4 * step,
                                 validation=metrics, config_sha256=config_hash), output / "best_validation_checkpoint.pth")
        if step % args.save_every == 0 or step % args.validate_every == 0 or step == final_step:
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, matrices=4 * step,
                             best_score=best_score, config_sha256=config_hash, rng_cpu=torch.get_rng_state(),
                             rng_device=torch.cuda.get_rng_state(device) if device.type == "cuda" else None), output / "last.pth")
        if step % args.log_every == 0 or step == start_step + 1 or step == final_step or "validation" in event:
            with (output / "history.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(event, draws=draws), allow_nan=False) + "\n")
            print(json.dumps(event, allow_nan=False), flush=True)
    print(json.dumps(dict(status="complete" if final_step == args.steps else "stopped", step=final_step,
                          matrices=4 * final_step, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
