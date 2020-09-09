# The basic configuration system
import os.path as osp
import sys

# Add src directory into the path
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../")))

import deeplab.config.deeplab_v3_plus

from yacs.config import CfgNode as CN

_C = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# --------------------------------------------------------------------------- #
# General Configuration
# --------------------------------------------------------------------------- #
# Set the random seed of the network for reproducibility
_C.RNG_SEED = -1

# We will create a sub-folder with this name in the output directory
# Unless otherwise specified, the output of our algorithm will be stored in [project root directory]/outputs/[TASK_NAME]
_C.TASK_NAME = "cfn_mtx_with_intensity"
_C.OUTPUT_DIR = ""  # Note that this directory does not need to contain the [TASK_NAME], we will add this for you.

# Ground truth semantic map label directory
_C.GROUND_TRUTH_DIR = ""

# The associate index of each label in the semantic segmentation network
_C.LABELS = [2, 1, 8, 10, 3]
# The name of the label
_C.LABELS_NAMES = ["road", "crosswalk", "lane", "vegetation", "sidewalk"]
# The RGB color of each label. We will use this to identify the label of each RGB pixel
_C.LABEL_COLORS = [
    [128, 64, 128],  # road
    [140, 140, 200],  # crosswalk
    [255, 255, 255],  # lane
    [107, 142, 35],  # vegetation
    [244, 35, 232],  # sidewalk
]

# If the time stamp reaches this threshold, we will stop generating the map.
# Usually, our start time frame is 390. If you want a shorter test time, you can set it to 1581541270, which is about
# 20 seconds.
_C.TEST_END_TIME = 1581541450

# --------------------------------------------------------------------------- #
# Mapping Node
# --------------------------------------------------------------------------- #
_C.MAPPING = CN()

# The source of the point cloud. It can be
# "dense_pcd" - The dense point cloud
# "raw_pcd" - The real time point cloud coming from the LiDAR sensor
_C.MAPPING.POINT_CLOUD_SOURCE = "dense_pcd"

# The resolution of the occupancy grid in meters
_C.MAPPING.RESOLUTION = 0.1
# The boundary of the occupancy grid, in meters. The format of the boundary is [[xmin, xmax], [ymin, ymax]]
_C.MAPPING.BOUNDARY = [[100, 300], [800, 1000]]

# Point cloud setting
_C.MAPPING.PCD = CN()
# If True, use the point cloud intensity data to augment our semantic BEV estimation
_C.MAPPING.PCD.USE_INTENSITY = True
# The maximum range of the point cloud, any point that is beyond this will be dropped.
_C.MAPPING.PCD.RANGE_MAX = 100.0

_C.MAPPING.CONFUSION_MTX = CN()
# The load path of the confusion matrix
_C.MAPPING.CONFUSION_MTX.LOAD_PATH = ""
# The store and load path of deterministic input to the mapping process
_C.MAPPING.INPUT_DIR = ""
# If round to close or round down
_C.MAPPING.ROUND_CLOSE = True

# --------------------------------------------------------------------------- #
# Semantic Segmentation Node
# --------------------------------------------------------------------------- #
_C.SEM_SEG = CN()

# Determine the scale of the input image, from 0 to 1.
_C.SEM_SEG.IMAGE_SCALE = 0.1

# Define the camera that we are interested in
# _C.SEM_SEG.TARGET_CAMERAS = ["camera1", "camera6"]
_C.SEM_SEG.TARGET_CAMERAS = ["camera1"]

# Network configuration
network_cfg = deeplab.config.deeplab_v3_plus.get_cfg_defaults()

network_cfg.DATASET.NAME = "Mapillary"
# The configuration file of the data set, it is useful for understanding the association between semantic labels and
# the classes.
network_cfg.DATASET.CONFIG_PATH = "/mnt/avl_shared/qinru/iros2020/resnext50_os8/config.json"
network_cfg.DATASET.IN_CHANNELS = 3
network_cfg.DATASET.NUM_CLASSES = 19

# The default model setting is based on resnext50_os8
# TODO: replace the default to the new trained resnext50
network_cfg.MODEL.WEIGHT = "/mnt/avl_shared/qinru/iros2020/resnext50_os8/run1/model_best.pth"
network_cfg.MODEL.SYNC_BN = False  # We don't need synchronized batch norm during inference time
network_cfg.MODEL.BACKBONE.NAME = "resnext50_32x4d"
network_cfg.MODEL.OUTPUT_STRIDE = 8
network_cfg.MODEL.ASPP.USE_DEPTHWISE_CNN = True
network_cfg.MODEL.DECODER.LOW_LEVEL_OUT_CHANNELS = 256
network_cfg.MODEL.DECODER.USE_DEPTHWISE_CNN = True

_C.SEM_SEG.NETWORK = network_cfg
