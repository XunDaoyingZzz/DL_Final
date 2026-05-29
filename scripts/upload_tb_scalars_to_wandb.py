#!/usr/bin/env python3
"""Upload TensorBoard scalar events to a W&B run."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from tensorboard.backend.event_processing import event_accumulator


def load_scalars(logdir: Path) -> dict[int, dict[str, float]]:
    if logdir.is_file() and logdir.name.startswith("events.out.tfevents"):
        event_files = [logdir]
    else:
        event_files = sorted(logdir.rglob("events.out.tfevents*"))

    if not event_files:
        raise SystemExit(f"No TensorBoard event files found under {logdir}")

    rows: dict[int, dict[str, float]] = {}
    for event_file in event_files:
        acc = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={event_accumulator.SCALARS: 0},
        )
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for scalar in acc.Scalars(tag):
                rows.setdefault(scalar.step, {})[tag] = scalar.value
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--project", default="hw3-3d-assets")
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument("--job-type", default="train")
    args = parser.parse_args()

    rows = load_scalars(args.logdir)
    run = wandb.init(
        project=args.project,
        name=args.name,
        group=args.group,
        job_type=args.job_type,
        sync_tensorboard=False,
    )
    for step in sorted(rows):
        wandb.log(rows[step], step=step)
    run.finish()
    print(f"uploaded {len(rows)} scalar steps from {args.logdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
