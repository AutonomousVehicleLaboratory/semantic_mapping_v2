from __future__ import absolute_import
import torch.nn as nn

from deeplab_v3_plus.models.deeplab_v3_plus import DeepLabV3Plus
from deeplab_v3_plus.models.loss import CrossEntropyLoss
from deeplab_v3_plus.models.metrics import MeanIOU


def build_xception(cfg):
    pass


def build_deeplabv3_plus(cfg):
    net = DeepLabV3Plus(in_channels=cfg.DATASET.IN_CHANNELS,
                        out_channels=cfg.DATASET.NUM_CLASSES,
                        backbone=cfg.MODEL.BACKBONE,
                        aspp_cfg=cfg.MODEL.ASPP,
                        decoder_cfg=cfg.MODEL.DECODER,
                        output_stride=cfg.MODEL.OUTPUT_STRIDE)
    loss_fn = CrossEntropyLoss(ignore_index=255)
    train_metric = MeanIOU(cfg.DATASET.NUM_CLASSES)
    val_metric = MeanIOU(cfg.DATASET.NUM_CLASSES)

    return net, loss_fn, train_metric, val_metric


def build_dummy_model(cfg):
    from core.nn.modules import Conv2d
    import torch.nn as nn
    import torchvision.models as models

    net = nn.Sequential(
        # *list(models.resnet101(pretrained=True).children())[:-2],  # We don't want the average pool and fc
        Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, bn=True),
        nn.Upsample(size=(16, 16), mode='bilinear', align_corners=True),
        Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, bn=True),
        nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True),
        Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bn=True),
        nn.Upsample(size=(513, 513), mode='bilinear', align_corners=True),
        Conv2d(in_channels=512, out_channels=cfg.DATASET.NUM_CLASSES, kernel_size=1)
    )

    # Initialize weight
    from core.nn.init import kaiming_normal
    for module in net:
        if isinstance(module, Conv2d):
            module.init_weights(kaiming_normal)

    # Ignore the segmentation boundary (which has index 255)
    loss_fn = CrossEntropyLoss(ignore_index=255)
    train_metric = MeanIOU(cfg.DATASET.NUM_CLASSES)
    val_metric = MeanIOU(cfg.DATASET.NUM_CLASSES)

    return net, loss_fn, train_metric, val_metric


# All the builder of models should be registered in _MODEL_BUILDERS
_MODEL_BUILDERS = {
    'Xception': build_xception,
    'DeepLabv3+': build_deeplabv3_plus,
    'Dummy': build_dummy_model,
}


def build_model(cfg):
    """General building function"""
    network, loss_fn, train_metric, val_metric = _MODEL_BUILDERS[cfg.MODEL.TYPE](cfg)

    if cfg.MODEL.SYNC_BN:
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network)

    return network, loss_fn, train_metric, val_metric
