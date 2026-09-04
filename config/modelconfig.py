import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import yaml
import os
import torch


@dataclass
class ModelConfig:
    """Training, evaluation and path configuration."""
    # Identifier shared by the preprocess, train, test and benchmark stages.
    runId: str
    model: str

    # Data parameters
    dataset: str
    chromosome: int
    population: str   # "ALL"/"AA"/"CA" or other race abbreviations
    segLen: int
    overlap: int

    testset: str = field(init=True, default=None)
    testPopulation: str = field(init=True, default=None)

    useWandB: bool = False
    WandBKey: str = field(default_factory=lambda: os.environ.get('WANDB_API_KEY', ''))
    WandBProjName: str = 'SCDA'
    unique: bool = False              # drop duplicate training segments

    # Model parameters
    vocabSize: int = 6
    padId: int = 5
    padLabel: int = 0                 # position label of a padding token
    missingId: int = 0
    dropoutRate: float = 0.05
    depth: int = 6
    stride: int = 1
    padding: int = 3
    kernelSize: int = 7
    nchannels: int = 48
    bioAware: bool = False
    bioEncoding: str = 'normPos'              # 'normPos'|'cM_gap'|'cM_norm'|'maf'|'cM_gap+maf'|...
    bioFile: str = field(default_factory=lambda: os.environ.get('BIO_FILE', './data/bio/chr22_bio.npz'))
    bioChannels: int = 1                       # number of bio channels (auto-derivable from bioEncoding)
    # Inject the biological prior as per-block scale and shift instead of an input channel.
    useFiLM: bool = False
    # Learned prototype set the bottleneck latent attends to.
    useCodebook: bool = False
    codebookSize: int = 256
    codebookNumHeads: int = 4
    codebookDropout: float = 0.0

    # Training parameters
    mixedPrecisionTraining: bool = False
    enableProfiling: bool = False
    maxProfilingBatches: int = 10
    saveCheckpointFreq: int = 1
    valFreq: int = 1                  # run validation every N epochs (and on the final epoch)
    useTorchCompile: bool = False     # compile the model with torch.compile()
    torchCompileMode: str = "reduce-overhead"  # see torch.compile docs
    valSubsetFraction: float = 1.0    # fraction of val batches to use each valFreq epoch
    useHaploidHeads: bool = False     # predict two binary alleles instead of the four-class genotype
    earlyStoppingPatience: int = 0    # 0 = disabled; else stop after N val epochs without val_loss improvement
    earlyStoppingMinDelta: float = 0.0  # min val_loss reduction to count as improvement
    curriculumStartRatio: float = -1.0  # anneal missingRatio from this value; negative disables
    curriculumWarmupEpochs: int = 0     # number of epochs over which to anneal (0 = no curriculum)
    useGroupNorm: bool = False          # add GroupNorm after each convolution
    gnGroups: int = 8                   # num_groups for GroupNorm
    batchSize: int = 64
    learningRate: float = None
#    device: str = 'gpu'
    # Scheduler-related
    scheduler: str = 'cosineAnn'  # "cosineAnn" or "stepLR"
    totalEpochs: int = 300
    warmupEpochs: int = 8
    cooldownEpochs: int = 0
    schedulerStepSize: int = 10
    schedulerGamma: float = 0.1

    # Loss-related
    loss: str = 'hybridWeightedFocalLoss'  # "focalLoss"/"hybridFocalLoss"/"hybridWeightedFocalLoss"/"weightedFocalLoss"/"crossEntropy"/"f1Loss"/...

    # Focal Loss parameters
    focalAlpha: float = 0.25
    focalGamma: float = 2.0

    # Weighted Focal Loss parameters
    weightedFocalGamma: float = 2.0

    # Hybrid Weighted Focal Loss parameters
    dosageLossLambda: float = 0.1
    # Add the validation split to the training data.
    trainOnVal: bool = False
    # Per-rank cap on positions per batch for segLen=-1 training. The bucket batch
    # size becomes maxBatchPositions // seq_len, clamped to [4, batchSize]. 0 disables.
    maxBatchPositions: int = 0
    # Comma-separated glob(s) of additional training HDF5 files.
    extraTrainFiles: str = ''
    # Score only the markers the reference-based imputer retained. False scores the
    # full target axis and flags those cells _FA.
    overlappedOnly: bool = True
    # Score only the markers the reference-based imputer could not match. Flags cells _EX.
    excludedOnly: bool = False
    # Index into `missing` to score. None scores every rate.
    missingLevelIdx: Optional[int] = None
    # DataLoader workers. 0 uses one per allocated CPU less one.
    dataLoaderWorkers: int = 0
    useLabelSmoothing: bool = False
    labelSmoothingGamma: float = 0.1
    normalizePenalties: bool = False

    # F1 Loss parameter
    f1Epsilon: float = 1e-7

    # Sampling-related
    sampling: str = 'normal'
    upsamplingRatio: float = 0.8

    # Finetuning, masking, etc.
    finetuning: bool = False
    targetBin: str = '0.001-0.005'
    missingRatio: float = 0.05
    dynamicRatio: bool = False
    bertStrategy: bool = False
    # Masking pattern during training: 'sporadic', 'colwise' or 'chip'.
    maskMode: str = 'sporadic'
    # Sample the per-batch colwise rate uniformly in (colwiseRateMin, missingRatio].
    colwiseDynamicRange: bool = False
    # Lower bound of that draw.
    colwiseRateMin: float = 0.0
    # Keep only columns above this in-batch minor allele frequency. 0 disables.
    colwiseMafMin: float = 0.0
    # Array site list (chrom, pos, maf) read when maskMode='chip'.
    chipSitesFile: str = ''
    # Load each HDF5 chunk into RAM instead of reading it per item.
    preloadData: bool = False

    # AdamW parameters
    adamwBeta1: float = 0.9
    adamwBeta2: float = 0.99
    adamwEps: float = 1e-08
    adamwWeightDecay: float = 0.001

    # Random seed
    seed: int = 0

    # Testing parameters
    benchmarkAll: bool = True
    benchmarkVariantsOnly: bool = False  # score only positions whose truth dosage is above zero
    perVariantR2: bool = False           # average per-variant Pearson r² instead of pooling
    numWorkers: int = 12
    epoch: int = None
    batchSizeTest: int = 1024


    # Directory paths
    resDir: str = "./res"
    modelDir: str = "./checkpoints"
    analysisDir: str = "./analysis"
    # Root of the dataset trees: split/, chunked/, masked/, subsets/.
    dataDir: str = "./data"

    # Optional job tracking
    jobId: Optional[str] = None
    run: Optional[str] = None

    # Computed fields
    bins: List[float] = field(
        default_factory=lambda: [
            0.001, 0.005, 0.010, 0.100, 0.200,
            0.300, 0.400, 0.500
        ]
    )
    # Computed fields
    missing: List[float] = field(
        default_factory=lambda: [
            0.05, 0.15, 0.25, 0.5
        ]
    )

    # Optional: restrict test.py to a subset of rand seeds for faster iteration.
    testRandStates: List[int] = field(default_factory=list)

    # Optional: list of chromosomes to use for *training* (multi-chromosome runs).
    # Empty → use [chromosome]. Test/eval always use the single `chromosome` field.
    chromosomes: List[int] = field(default_factory=list)

    padValueOneHot: List[int] = field(
        default_factory=lambda: [0, 0, 0, 0, 0, 1]
    )

    binRanges: List[str] = field(default_factory=list)

    def __post_init__(self):
        """
        Initialize run name if not provided.
        """
        if self.testset is None:
            self.testset = self.dataset
        if self.testPopulation is None:
            self.testPopulation = self.population
        if not self.chromosomes:
            self.chromosomes = [self.chromosome]

        if not self.run:
            self.run = (
                f"{self.runId}_{self.dataset}_chr{self.chromosome}"
                f"_seg{self.segLen}_overlap{self.overlap}"
            )
        if not self.binRanges:  # Only calculate if binRanges is empty
            self.binRanges = [f"{self.bins[i]:.3f}-{self.bins[i+1]:.3f}" for i in range(len(self.bins) - 1)]


    @property
    def stepSize(self) -> int:
        """Step size when using StepLR as scheduler."""
        return int(0.3* self.totalEpochs)

    @property
    def padValueTensor(self) -> torch.Tensor:
        """Return padValueOneHot as a tensor."""
        return torch.tensor(self.padValueOneHot, dtype=torch.float32)

    @property
    def missing_percent_strs(self) -> List[str]:
        """Helper property for formatted missing percentages as strings."""
        return [f"{m * 100:.0f}%" for m in self.missing]

    @property
    def test_csv_gz(self) -> str:
        """Path to test CSV file"""
        return (
            f"{self.dataDir}/{self.dataset}/split/{self.dataset}_"
            f"chr{self.chromosome}_{self.population}_test.csv.gz"
        )

    @property
    def masked_csv_gzs(self):
        def get_paths(rand_state):
            return [
                f"{self.dataDir}/{self.dataset}/masked/{rand_state}/{m_str}/"
                f"{self.dataset}_chr{self.chromosome}_{self.population}_"
                f"missing{m_str}_masked.csv.gz"
                for m_str in self.missing_percent_strs
            ]
        return get_paths
    @property
    def imputed_csv_gzs(self):
        """Returns a function that generates paths to imputed data CSV files for all missing levels."""
        def get_paths(rand_state):
            return [
                f"./impute/{rand_state}/{m_str}/{self.runId}_epoch{self.epoch}_"
                f"{self.dataset}_chr{self.chromosome}_{self.population}_"
                f"seg{self.segLen}_overlap{self.overlap}_missing{m_str}_imputed.csv.gz"
                for m_str in self.missing_percent_strs
            ]
        return get_paths
    @property
    def test_dataset_file(self) -> str:
        """Path to test dataset HDF5 file"""
        return os.path.join(
            self.resDir,
            f"{self.dataset}_chr{self.chromosome}_{self.population}_"
            f"seg{self.segLen}_overlap{self.overlap}_test.hdf5"
        )

    @property
    def masked_dataset_files(self) -> List[str]:
        """Paths to masked dataset HDF5 files for all missing levels."""
        def get_paths(rand_state):
            return [
                os.path.join(
                    self.resDir,
                    f"{self.dataset}_chr{self.chromosome}_{self.population}_"
                    f"seg{self.segLen}_overlap{self.overlap}_missing{m_str}_rand{rand_state}_masked.hdf5"
                )
                for m_str in self.missing_percent_strs
            ]
        return get_paths

    @property
    def checkpoint(self) -> str:
        """Checkpoint path"""
        return self._get_checkpoint_path("checkpoint")

    def _get_checkpoint_path(self, prefix: str) -> str:
        """Internal method to construct checkpoint paths"""
        return os.path.join(
            self.modelDir,
            f"{prefix}_{self.runId}_"
            f"{self.dataset}_chr{self.chromosome}_{self.population}_"
            f"seg{self.segLen}_overlap{self.overlap}_epoch_{self.epoch}.pth"
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ModelConfig':
        """Create a ModelConfig instance from a YAML file.

        Args:
        yaml_path: Path to the YAML configuration file

        Returns:
        ModelConfig: Configuration object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Fill in default values for missing fields
        for field_name, field_def in cls.__dataclass_fields__.items():
            if field_name not in config_dict:
                # Use the default value or the default_factory if available
                if field_def.default is not dataclasses.MISSING:
                    config_dict[field_name] = field_def.default
                elif field_def.default_factory is not dataclasses.MISSING:  # For fields with default_factory
                    config_dict[field_name] = field_def.default_factory()

        return cls(**config_dict)

    @classmethod
    def from_args(cls) -> 'ModelConfig':
        """Create a ModelConfig instance from command line arguments.

        Returns:
            ModelConfig: Configuration object
        """
        parser = argparse.ArgumentParser(
            description="Training Autoencoder on Genotype Data for Imputation"
        )

        # Required arguments
        parser.add_argument(
            '--runId',
            type=str,
            required=True,
            help="Unique run identifier for the training run in wandb.ai"
        )
        parser.add_argument(
            '--epoch',
            type=int,
            required=True,
            help="Specify the epoch for the checkpoint"
        )

        # Optional arguments - Data parameters
        parser.add_argument(
            '--dataset',
            type=str,
            default=None,
            help="Dataset name"
        )
        parser.add_argument(
            '--chromosome',
            type=int,
            default=None,
            help="Chromosome number"
        )
        parser.add_argument(
            '--segLen',
            type=int,
            default=None,
            help="Sample segment length"
        )
        parser.add_argument(
            '--overlap',
            type=int,
            default=None,
            help="Sample overlap"
        )

        # Model parameters
        parser.add_argument(
            '--vocabSize',
            type=int,
            default=6,
            help="SNPs' one-hot encoding dim"
        )
        parser.add_argument(
            '--padId',
            type=int,
            default=5,
            help="Sample padding value"
        )
        parser.add_argument(
            '--padLabel',
            type=int,
            default=-3,
            help="Column label for padding entries"
        )
        parser.add_argument(
            '--missingId',
            type=int,
            default=0,
            help="Sample padding value"
        )
        parser.add_argument(
            '--dropoutRate',
            type=float,
            default=None,
            help="Dropout rate"
        )
        parser.add_argument(
            '--depth',
            type=int,
            default=6,
            help="Encoder/decoder's depth"
        )
        parser.add_argument(
            '--stride',
            type=int,
            default=1,
            help="Convolutional stride"
        )
        parser.add_argument(
            '--padding',
            type=int,
            default=3,
            help="Model padding"
        )
        parser.add_argument(
            '--kernelSize',
            type=int,
            default=7,
            help="Kernel size"
        )
        parser.add_argument(
            '--nchannels',
            type=int,
            default=32,
            help="First layer's output number of channels"
        )

        # Testing parameters
        parser.add_argument(
            '--device',
            type=str,
            default=None,
            help="Training device (cpu/cuda)"
        )
        parser.add_argument(
            '--population',
            type=str,
            default=None,
            help="Population group"
        )
        parser.add_argument(
            '--missingRatio',
            type=float,
            default=None,
            help="Percentage of missing data"
        )
        parser.add_argument(
            '--benchmarkAll',
            type=bool,
            default=True,
            help="Benchmark on all data if True, else on missing data only"
        )
        parser.add_argument(
            '--numWorkers',
            type=int,
            default=None,
            help="Number of CPU workers for dataloader"
        )
        parser.add_argument(
            '--batchSize',
            type=int,
            default=128,
            help="Batch size for training"
        )
        parser.add_argument(
            '--unique',
            type=bool,
            default=True,
            help="Remove duplication from the SNPs segments"
        )

        # Directory paths
        parser.add_argument(
            '--resDir',
            type=str,
            default='./res',
            help="Directory for results"
        )
        parser.add_argument(
            '--modelDir',
            type=str,
            default='./checkpoints',
            help="Directory for saved models"
        )
        parser.add_argument(
            '--learningRate',
            type=float,
            default=None,
            help="Learning rate for training"
        )
        parser.add_argument(
            '--loss',
            type=str,
            default='focalLoss',
            help="Loss function type"
        )
        parser.add_argument(
            '--sampling',
            type=str,
            default='upsampling',
            help="Sampling strategy"
        )
        parser.add_argument(
            '--upsamplingRatio',
            type=float,
            default=0.8,
            help="Ratio for upsampling"
        )
        parser.add_argument(
            '--finetuning',
            type=bool,
            default=False,
            help="Whether to perform finetuning"
        )
        parser.add_argument(
            '--targetBin',
            type=str,
            default='0.000-0.005',
            help="Target bin for training"
        )
        parser.add_argument(
            '--totalEpochs',
            type=int,
            default=200,
            help="Total number of training epochs"
        )
        parser.add_argument(
            '--warmupEpochs',
            type=int,
            default=5,
            help="Number of warmup epochs"
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help="Random seed"
        )
        parser.add_argument(
            '--useWandB',
            type=bool,
            default=False,
            help="Whether to use Weights & Biases"
        )
        parser.add_argument(
            '--WandBKey',
            type=str,
            default=os.environ.get('WANDB_API_KEY', ''),
            help="Weights & Biases API key (default from $WANDB_API_KEY)"
        )
        parser.add_argument(
            '--WandBProjName',
            type=str,
            default='SCDA',
            help="Weights & Biases project name"
        )
        parser.add_argument(
            '--batchSizeTest',
            type=int,
            default=1024,
            help="Batch size for testing"
        )
        parser.add_argument(
            '--analysisDir',
            type=str,
            default='./analysis',
            help="Directory for analysis outputs"
        )
        # Optional job tracking
        parser.add_argument(
            '--jobId',
            type=str,
            default=None,
            help="Eval job ID"
        )

        args = parser.parse_args()
        return cls(**vars(args))

    def save_yaml(self, yaml_path: str):
        """Save current configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML file
        """
        # Convert the config to a dictionary, excluding computed properties
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and not isinstance(v, (torch.Tensor, property))
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

