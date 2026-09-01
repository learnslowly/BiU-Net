"""Generate train configs at the original manuscript seg128/overlap16
settings (HLA uses overlap=64 per original).
"""
from pathlib import Path

CFG_DIR = Path("configs")

# (dataset, chromosome, population, segLen, overlap)
DATASETS = [
    ("1KGP", 22, "ALL", 128, 16),
    ("SGDP", 22, "ALL", 128, 16),
    ("HLA",  6,  "ALL", 128, 64),
    ("LOS",  22, "ALL", 128, 16),
    ("LOS",  22, "AA",  128, 16),
    ("LOS",  22, "CA",  128, 16),
]

TEMPLATE = """runId: exp1
dataset: {dataset}
chromosome: {chrom}
chromosomes: [{chrom}]
segLen: {segLen}
overlap: {overlap}
population: {pop}
model: unet
bioAware: True
depth: 6
nchannels: 48
kernelSize: 7
stride: 1
dropoutRate: 0.05
useGroupNorm: False
batchSize: 256
learningRate: 0.0005
loss: hybridWeightedFocalLoss
dosageLossLambda: 0.1
focalAlpha: 0.25
focalGamma: 2.0
sampling: normal
missingRatio: 0.30
dynamicRatio: True
bertStrategy: False
benchmarkAll: True
mixedPrecisionTraining: True
useTorchCompile: True
torchCompileMode: reduce-overhead
saveCheckpointFreq: 50
valFreq: 10
valSubsetFraction: 0.5
useWandB: False
finetuning: False
totalEpochs: 700
warmupEpochs: 10
cooldownEpochs: 0
scheduler: cosineAnn
adamwBeta1: 0.9
adamwBeta2: 0.99
adamwEps: 0.00000001
adamwWeightDecay: 0.01
seed: 42
earlyStoppingPatience: 40
earlyStoppingMinDelta: 0.0001
curriculumStartRatio: 0.5
curriculumWarmupEpochs: 200
bins: [0.001, 0.005, 0.010, 0.100, 0.200, 0.300, 0.400, 0.500]
batchSizeTest: 8
epoch: 0
"""

for ds, chrom, pop, segLen, overlap in DATASETS:
    body = TEMPLATE.format(dataset=ds, chrom=chrom, pop=pop, segLen=segLen, overlap=overlap)
    out = CFG_DIR / f"train_{ds}_chr{chrom}_{pop}_seg{segLen}.yaml"
    out.write_text(body)
    print(f"Wrote {out}")
