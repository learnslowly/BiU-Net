"""Dry-run a training config: resolve every input it will open, open none of them.

Usage: python scripts/preflight_train_config.py <config.yaml>
Exits non-zero on the first problem so a batch can gate on it.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.modelconfig import ModelConfig
from data.utils import get_dataset_paths

if len(sys.argv) < 2:
    sys.exit(__doc__)
cfg_path = sys.argv[1]
config = ModelConfig.from_yaml(cfg_path)

print(f"config      : {cfg_path}")
print(f"runId       : {config.runId}")
print(f"dataset     : {config.dataset}  chromosomes {config.chromosomes}")
print(f"segLen/ovlp : {config.segLen}/{config.overlap}")
print(f"batchSize   : {config.batchSize} per rank")
print(f"lr / epochs : {config.learningRate} / {config.totalEpochs}")
print(f"curriculum  : {config.curriculumStartRatio} -> {config.missingRatio} "
      f"over {config.curriculumWarmupEpochs} epochs")
print(f"earlyStop   : patience {config.earlyStoppingPatience}")
print(f"trainOnVal  : {config.trainOnVal}   preload: {config.preloadData}")

problems = []
# get_dataset_paths already folds in trainOnVal and extraTrainFiles, and raises
# if the extra glob matches nothing -- so this call is the real gate.
train_files, val_files = get_dataset_paths(config)
for name, files in (("train", train_files), ("val", val_files)):
    missing = [f for f in files if not os.path.exists(f)]
    print(f"{name:11s}: {len(files)} files, {len(missing)} missing")
    if not files:
        problems.append(f"{name} file list is empty")
    problems += [f"{name} file missing: {f}" for f in missing[:5]]
    cached = sum(1 for f in files if os.path.exists(f + ".compact.npz"))
    if config.preloadData:
        print(f"{'':11s}  compact cache warm for {cached}/{len(files)}")

extra = getattr(config, "extraTrainFiles", "") or ""
if extra:
    hits = sorted(glob.glob(extra))
    print(f"extraTrain : glob {extra} -> {len(hits)} files")
    if not hits:
        problems.append(f"extraTrainFiles glob matched nothing: {extra}")
    cached = sum(1 for f in hits if os.path.exists(f + ".compact.npz"))
    if config.preloadData:
        print(f"{'':11s}  compact cache warm for {cached}/{len(hits)}")

ckpts = sorted(glob.glob(os.path.join(
    config.modelDir, f"checkpoint_{config.runId}_*_epoch_*.pth")))
print(f"checkpoints: {len(ckpts)} present for this runId"
      + (f" (latest {os.path.basename(ckpts[-1])})" if ckpts else " -> random init"))

if problems:
    print("\nPREFLIGHT FAILED:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("\nPREFLIGHT OK")
