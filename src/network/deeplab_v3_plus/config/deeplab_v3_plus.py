from __future__ import absolute_import
from src.network.deeplab_v3_plus.config.base import CN, _C

# Create public alias
cfg = _C

# Input channels
_C.DATASET.IN_CHANNELS = 3
_C.DATASET.NUM_CLASSES = 21

_C.MODEL.TYPE = "DeepLabv3+"
# Backbone used in DeepLabv3+
_C.MODEL.BACKBONE = "resnet"
_C.MODEL.OUTPUT_STRIDE = 16

# --------------------------------------------------------------------------- #
# ASPP
# --------------------------------------------------------------------------- #
_C.MODEL.ASPP = CN()

_C.MODEL.ASPP.OUT_CHANNELS = 256
_C.MODEL.ASPP.ATROUS_CHANNELS = [256, 256, 256, 256]
_C.MODEL.ASPP.ATROUS_KERNEL_SIZE = [1, 3, 3, 3]
_C.MODEL.ASPP.ATROUS_DILATION = [1, 6, 12, 18]
_C.MODEL.ASPP.DROPOUT = 0.5

# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
_C.MODEL.DECODER = CN()

_C.MODEL.DECODER.LOW_LEVEL_OUT_CHANNELS = 48
_C.MODEL.DECODER.REFINE_CHANNELS = [256, 256]
_C.MODEL.DECODER.REFINE_KERNEL_SIZE = [3, 3]
