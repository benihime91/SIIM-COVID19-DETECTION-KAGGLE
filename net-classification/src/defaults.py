from yacs.config import CfgNode as CN
from pytorch_toolbelt.utils.random import get_random_name


CPUS = 8

NUM_EPOCHS = 20
TOTAL_BATCH = 32

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.RUN_NAME = get_random_name()
_C.CHECKPOINT = " "
_C.SEED = 42

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "runs/"
# path to csv directory
_C.PATH_CSV = "/home/stud1901/work/data/train_stratified_group_5xfolds_clean.csv"
# path where the images are stored
_C.PATH_IMAGES = "/home/stud1901/work/data/images/train/"
# path where the segmented masks are stored
_C.PATH_MASK = "/home/stud1901/work/data/images/mask"
_C.PATH_PSEUDO = None  # Path to the pseudo labels
_C.NOISY_CSV = None

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the images during training
_C.INPUT.INPUT_SIZE = 512
# Number of output classes
_C.INPUT.NUM_CLASSES = 4
# Number of channels
_C.INPUT.NUM_CHANNELS = 3
_C.INPUT.DRAW_OVERLAY = False
# Training augmentations
_C.INPUT.TRAIN_AUGS = " "


# -----------------------------------------------------------------------------
# Training Options
# -----------------------------------------------------------------------------
_C.TRAINING = CN()
_C.TRAINING.EMA = False
# adds a segmentation auxilliary loss
_C.TRAINING.AUX_LOSS = True
_C.TRAINING.AUX_LOSS_WEIGHT = 1.0
# optimizer options
_C.TRAINING.OPTIMIZER = CN()
# name of the optimizer
_C.TRAINING.OPTIMIZER.NAME = "Ranger21"
# learning rate, weight decay; init options for optimizer
_C.TRAINING.OPTIMIZER.LR = 1e-04
_C.TRAINING.OPTIMIZER.WEIGHT_DECAY = 1e-02
_C.TRAINING.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAINING.OPTIMIZER.EPS = 1e-08
# Extra init args for the optimizer
_C.TRAINING.OPTIMIZER.INIT_ARGS = CN(
    {"use_adaptive_gradient_clipping": False, "use_madgrad": False,}
)

# Use SAM optimizer
_C.TRAINING.OPTIMIZER.USE_SAM = False

# LR Scheduler name
_C.TRAINING.SCHEDULER = CN()
_C.TRAINING.SCHEDULER.NAME = None
# LR scheduler options; Scheduler specific
_C.TRAINING.SCHEDULER.EPOCHS = NUM_EPOCHS
_C.TRAINING.SCHEDULER.WARMUP_EPOCHS = 0  # %
_C.TRAINING.SCHEDULER.WARMDOWN_PCT = 0.72
_C.TRAINING.SCHEDULER.MAX_LR = _C.TRAINING.OPTIMIZER.LR

# Loss func
_C.TRAINING.LOSS = CN()
_C.TRAINING.LOSS.CLASS_LOSS = "cross_entropy"
# Add a combination of Asymetric and focal loss to classification loss
_C.TRAINING.LOSS.ADD_CLASS_COMB = False


_C.TRAINING.LOSS.MASK_LOSS = "bce_loss"

# Segmentation Combo Loss func
_C.TRAINING.LOSS.COMBO_LOSS = CN()
# use combo loss for image segmentation
_C.TRAINING.LOSS.COMBO_LOSS.USE = False
# Loss function 1 for combo loss
_C.TRAINING.LOSS.COMBO_LOSS.L1 = "lovasz_loss"
# Loss function 2 for combo loss
_C.TRAINING.LOSS.COMBO_LOSS.L2 = "bce_loss"
# Weights for combo loss
_C.TRAINING.LOSS.COMBO_LOSS.WEIGHTS = [0.75, 0.25]


_C.MODEL = CN()
_C.MODEL.USE_MISH = False
# ---------------------------------------------------------------------------- #
# Model Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "efficientnetb4"
_C.MODEL.BACKBONE.DROP_PATH = 0.5
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.BACKBONE.INIT_ARGS = CN({"global_pool": "", "num_classes": 0})
# ---------------------------------------------------------------------------- #
# Model Classification Head
# ---------------------------------------------------------------------------- #
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "lin"
_C.MODEL.HEAD.POOLING = "avg"
_C.MODEL.HEAD.DROPOUT = 0.5
_C.MODEL.HEAD.ATTENTION_MAP = None
_C.MODEL.HEAD.MULTI_DROPOUT = CN()
_C.MODEL.HEAD.MULTI_DROPOUT.USE = False
_C.MODEL.HEAD.MULTI_DROPOUT.NUM = 5
# ---------------------------------------------------------------------------- #
# Model Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEG_HEAD = CN()
_C.MODEL.SEG_HEAD.LIN_FTRS = 128
_C.MODEL.SEG_HEAD.RESIDUAL = False
# ---------------------------------------------------------------------------- #
# Auxiliary Model
# ---------------------------------------------------------------------------- #
_C.MODEL.AUX_MODEL = CN()
_C.MODEL.AUX_MODEL.VERSION = "v1"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Mini batch size
_C.DATALOADER.BATCH_SIZE = 32
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = CPUS

# ---------------------------------------------------------------------------- #
# Trainer
# ---------------------------------------------------------------------------- #
_C.TRAINER = CN()
_C.TRAINER.accumulate_grad_batches = TOTAL_BATCH // _C.DATALOADER.BATCH_SIZE
_C.TRAINER.terminate_on_nan = True
_C.TRAINER.num_sanity_val_steps = 0
_C.TRAINER.precision = 16
_C.TRAINER.log_every_n_steps = 5
_C.TRAINER.max_epochs = NUM_EPOCHS
_C.TRAINER.gpus = 1
_C.TRAINER.gradient_clip_val = 0.0
_C.TRAINER.gradient_clip_algorithm = "norm"
_C.TRAINER.stochastic_weight_avg = False
_C.TRAINER.limit_train_batches = 1.0


def get_cfg() -> CN:
    """
    Get a copy of the default config.
    """
    return _C.clone()


if __name__ == "__main__":
    import pprint

    pprint.pprint(_C)

