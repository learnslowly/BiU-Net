"""Seed a fine-tuning run with the weights of the genome-wide model.

Usage:
  python scripts/prebake_finetune_checkpoint.py SOURCE_CONFIG TARGET_CONFIG

  SOURCE_CONFIG   the genome-wide training config whose checkpoint to copy
  TARGET_CONFIG   the fine-tuning training config to seed
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.modelconfig import ModelConfig  # noqa: E402
from data.utils import find_latest_checkpoint  # noqa: E402


def run_name(config):
    """The checkpoint naming pattern train.py builds before it resumes."""
    return (f"{config.runId}_{config.dataset}_chr{config.chromosome}"
            f"_{config.population}_seg{config.segLen}_overlap{config.overlap}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.split("Usage:")[1].strip())
    source = ModelConfig.from_yaml(sys.argv[1])
    target = ModelConfig.from_yaml(sys.argv[2])

    # the source checkpoint is the genome-wide model's own latest
    source.run = run_name(source)
    src_path = find_latest_checkpoint(source)
    if src_path is None:
        raise SystemExit(
            f"no checkpoint for {source.runId} under {source.modelDir}; "
            "train the genome-wide model before seeding a fine-tuning run")

    target.run = run_name(target)
    dst_path = os.path.join(target.modelDir, f"checkpoint_{target.run}_epoch_0.pth")
    if os.path.exists(dst_path):
        print(f"{dst_path} already exists, leaving it alone")
        return

    state = torch.load(src_path, map_location="cpu")
    seed = {"state_dict": state["state_dict"], "epoch": 0}
    os.makedirs(target.modelDir, exist_ok=True)
    torch.save(seed, dst_path)
    print(f"seeded {target.runId} chromosome {target.chromosome} "
          f"from {os.path.basename(src_path)}")


if __name__ == "__main__":
    main()
