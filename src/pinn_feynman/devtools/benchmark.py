"""Time training steps of a configuration on a device (CPU/GPU comparison).

    python -m pinn_feynman.devtools.benchmark configs/box_massless/baseline.yaml --device cuda --dtype float32 --batch-size 64 --steps 300

Prints one JSON line: device, dtype, batch size, threads, milliseconds per step, and the
implied minutes for one full training (epochs x batches per epoch). Uses the cached
boundary and validation datasets of the configuration.
"""

import argparse
import copy
import json
import time

import torch

from ..config import load_config
from ..training.setup import build_experiment


def time_steps(trainer, steps, warmup=20):
    """Mean seconds per optimisation step (after `warmup` steps), synchronising CUDA."""
    batches = trainer.sampler.epoch_indices()
    sync = torch.cuda.synchronize if trainer.device.type == "cuda" else (lambda: None)
    for i in range(warmup):
        trainer.train_step(batches[i % len(batches)])
    sync()
    start = time.perf_counter()
    for i in range(steps):
        trainer.train_step(batches[i % len(batches)])
    sync()
    return (time.perf_counter() - start) / steps


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default=None, help="override the configuration dtype")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)

    config = copy.deepcopy(load_config(args.config))
    config["training"]["device"] = args.device
    if args.dtype:
        config["dtype"] = args.dtype
    if args.batch_size:
        col = config["collocation"]
        batches = col["n_per_epoch"] // col["batch_size"]          # keep the number of steps per epoch
        col["batch_size"] = args.batch_size
        col["n_per_epoch"] = min(args.batch_size * batches, col["n_base"] // args.batch_size * args.batch_size)
    torch.set_num_threads(args.threads)
    trainer = build_experiment(config, "real").trainer
    seconds = time_steps(trainer, args.steps)
    col = config["collocation"]
    steps_per_epoch = col["n_per_epoch"] // col["batch_size"]
    result = {"device": args.device, "dtype": config["dtype"], "batch_size": col["batch_size"],
              "threads": args.threads, "ms_per_step": 1000 * seconds, "steps_per_epoch": steps_per_epoch,
              "minutes_per_training": seconds * steps_per_epoch * config["training"]["epochs"] / 60,
              "torch": torch.__version__}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    main()
