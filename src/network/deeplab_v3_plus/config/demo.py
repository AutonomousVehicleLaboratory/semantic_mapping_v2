"""
Configuration for running the demo in AVL dataset
"""
from __future__ import absolute_import
from yacs.config import CfgNode as CN
from src.network.deeplab_v3_plus.config.deeplab_v3_plus import _C as DEEPLAB_CN

_C = CN()
# Create public alias
cfg = _C

# if set to @, the filename of config will be used by default
_C.OUTPUT_DIR = '@'
# Name of the output video. It will be saved in the output directory
# a surfix of .avi will be added into the video name
_C.OUTPUT_NAME = ""
# Training dataset
_C.TRAIN_DATASET = ""
_C.DATASET_CONFIG = ""
# ================================================== #
# Dataset
# ================================================== #
_C.DATASET = CN()
_C.DATASET.NAME = ''
# Number of channels in the image
_C.DATASET.IN_CHANNELS = 0
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.ROOT_DIR = ""
# ================================================== #
# Neural Network
# ================================================== #
_C.MODEL = CN()
_C.MODEL.TYPE = ''
# Path to Pre-trained or checkpointed weights
_C.MODEL.WEIGHT = ''
# When set to True, the model will use synchronized batch normalization
_C.MODEL.SYNC_BN = False
# ================================================== #
# Deeplab specific setting
# ================================================== #
_C.MODEL.BACKBONE = DEEPLAB_CN.MODEL.BACKBONE
_C.MODEL.OUTPUT_STRIDE = DEEPLAB_CN.MODEL.OUTPUT_STRIDE
_C.MODEL.ASPP = DEEPLAB_CN.MODEL.ASPP
_C.MODEL.DECODER = DEEPLAB_CN.MODEL.DECODER
