"""Basic experiments configuration

This file is the one-stop reference point for all configurable options. Typically we will create one YAML configuration
file for each experiment. Each configuration file only overrides (or extend) the options that are changing (adding) in
that experiment.
"""

from yacs.config import CfgNode as CN

# --------------------------------------------------------------------------- #
# Constant
# --------------------------------------------------------------------------- #
DISABLE = 0

# --------------------------------------------------------------------------- #
# Basic
# --------------------------------------------------------------------------- #
_C = CN()

# Name of the task
# It can be arbitrary name with special characters
# It is mainly for the name of logger and output files
_C.TASK_NAME = ''
# Automatically resume weights from last checkpoints
_C.AUTO_RESUME = True
# Whether to resume the optimizer and the scheduler
_C.RESUME_STATES = True

# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.TYPE = ''
# Path to Pre-trained or checkpointed weights
_C.MODEL.WEIGHT = ''
# When set to True, the model will use synchronized batch normalization
_C.MODEL.SYNC_BN = False

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.NAME = ''
# Number of channels in the image
_C.DATASET.IN_CHANNELS = 0
_C.DATASET.NUM_CLASSES = 0

# Root directory of dataset
_C.DATASET.ROOT_DIR = ''
# Name of the split for training
_C.DATASET.TRAIN = ''
# Name of the split for validation
_C.DATASET.VAL = ''
# Name of the split for test
_C.DATASET.TEST = ''

# --------------------------------------------------------------------------- #
# DataLoader
# --------------------------------------------------------------------------- #
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 0
# Whether to drop last
_C.DATALOADER.DROP_LAST = True
# Enable automatic memory pinning for fast CPU to GPU data transfer
_C.DATALOADER.PIN_MEMORY = True

# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = ''

# Basic parameters of the optimizer
# Note that the learning rate should be changed according to batch size
_C.OPTIMIZER.BASE_LR = 0.001

_C.OPTIMIZER.WEIGHT_DECAY = 0.0

# Maximum norm of gradients. 0 for disable
_C.OPTIMIZER.MAX_GRAD_NORM = DISABLE

# Specific parameters of OPTIMIZERs
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.momentum = 0.0
_C.OPTIMIZER.SGD.dampening = 0.0
_C.OPTIMIZER.SGD.nesterov = False

_C.OPTIMIZER.Adam = CN()
_C.OPTIMIZER.Adam.betas = (0.9, 0.999)

# --------------------------------------------------------------------------- #
# Learning Rate Scheduler
# --------------------------------------------------------------------------- #
_C.SCHEDULER = CN()
_C.SCHEDULER.TYPE = ''

_C.SCHEDULER.MAX_EPOCH = 1
# Minimum learning rate. negative for disable.
_C.SCHEDULER.CLIP_LR = 0.0

# Specific parameters of SCHEDULERs
_C.SCHEDULER.StepLR = CN()
_C.SCHEDULER.StepLR.step_size = 0
_C.SCHEDULER.StepLR.gamma = 0.1

_C.SCHEDULER.MultiStepLR = CN()
_C.SCHEDULER.MultiStepLR.milestones = ()
_C.SCHEDULER.MultiStepLR.gamma = 0.1

_C.SCHEDULER.PolyLRDecay = CN()
_C.SCHEDULER.PolyLRDecay.max_iter = 300
_C.SCHEDULER.PolyLRDecay.power = 0.9

# --------------------------------------------------------------------------- #
# Specific train options
# --------------------------------------------------------------------------- #
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 1

# Period to save checkpoints. 0 for disable
_C.TRAIN.CHECKPOINT_PERIOD = DISABLE
# Period to log training status. 0 for disable
_C.TRAIN.LOG_PERIOD = DISABLE

# Data augmentation. The format is 'method' or ('method', *args)
# For example
# _C.TRAIN.AUGMENTATION = ('ToTensor', ('Normalize', (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), True),)
_C.TRAIN.AUGMENTATION = ()

# Regex patterns of modules and/or parameters to freeze
_C.TRAIN.FROZEN_PATTERNS = ()

# True if we freeze all batch normalization layers
_C.TRAIN.FREEZE_BATCHNORM = False

# --------------------------------------------------------------------------- #
# Specific validation options
# --------------------------------------------------------------------------- #
_C.VALIDATE = CN()

_C.VALIDATE.BATCH_SIZE = 1

# Period to validate. The unit here is epoch. 0 for disable.
_C.VALIDATE.PERIOD = DISABLE
# Period to log the validation. The unit here is iteration.
_C.VALIDATE.LOG_PERIOD = DISABLE
# The metric for best validation performance
_C.VALIDATE.METRIC = ''

_C.VALIDATE.AUGMENTATION = ()

# --------------------------------------------------------------------------- #
# Specific test options
# --------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1

_C.TEST.LOG_PERIOD = DISABLE

# The path of weights to be tested. '@' has similar syntax as OUTPUT_DIR.
# If not set, the last checkpoint will be used by default.
_C.TEST.WEIGHT = ''

# Data augmentation.
_C.TEST.AUGMENTATION = ()

# --------------------------------------------------------------------------- #
# Misc options
# --------------------------------------------------------------------------- #
# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'

# Set seed for reproducibility, Note that non-determinism may still be present
# due to non-deterministic operator implementations in GPU operator libraries
# -1 means the seed is not fixed. 
_C.RNG_SEED = -1
