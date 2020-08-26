from __future__ import absolute_import
import torch.nn as nn
import torch.nn.functional as F

from deeplab_v3_plus.models.backbone import build_backbone
from deeplab_v3_plus.models.aspp import AtrousSpatialPyramidPoolingModule
from deeplab_v3_plus.models.decoder import Decoder


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 backbone,
                 aspp_cfg,
                 decoder_cfg,
                 output_stride):
        super(DeepLabV3Plus, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_stride = output_stride

        self.backbone = build_backbone(backbone, output_stride)

        # Expand the atrous dilation (i.e. its receptive field) when output_stride is 8
        if output_stride == 16:
            atrous_dilation = [1, 6, 12, 18]
        elif output_stride == 8:
            atrous_dilation = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        self.aspp = AtrousSpatialPyramidPoolingModule(in_channels=self.backbone.out_channels,
                                                      out_channels=aspp_cfg.OUT_CHANNELS,
                                                      atrous_channels=aspp_cfg.ATROUS_CHANNELS,
                                                      atrous_kernel_size=aspp_cfg.ATROUS_KERNEL_SIZE,
                                                      atrous_dilation=atrous_dilation,
                                                      dropout=aspp_cfg.DROPOUT)

        self.decoder = Decoder(in_channels=self.aspp.out_channels,
                               out_channels=out_channels,
                               low_level_in_channels=self.backbone.low_level_channels,
                               low_level_out_channels=decoder_cfg.LOW_LEVEL_OUT_CHANNELS,
                               refine_channels=decoder_cfg.REFINE_CHANNELS,
                               refine_kernel_size=decoder_cfg.REFINE_KERNEL_SIZE)

    def forward(self, x, upsample_pred=True):
        """
        Args:
            x: Network input image
            upsample_pred: True if the network prediction is upsampled to the same dimension as input
        """
        input_size = x.shape[2:]

        feature_dict = self.backbone(x)
        feature = feature_dict["feature"]
        low_feature = feature_dict["low_feature"]

        feature = self.aspp(feature)
        feature = self.decoder(feature, low_feature)

        # As mentioned in the deeplabv3 paper, it is important to keep the groundtruths intact and
        # instead upsample the final logits.
        if upsample_pred:
            feature = F.interpolate(feature, size=input_size, mode='bilinear', align_corners=True)

        return feature


if __name__ == '__main__':
    # If you want to run this main() add following script to the top of this file
    #
    # import os.path as osp
    # import sys
    #
    # sys.path.insert(0, osp.dirname(__file__) + '/../..')

    import torch
    from deeplab_v3_plus.config.deeplab_v3_plus import cfg

    cfg.merge_from_file("/home/qinru/avl/semantic/deeplab/experiments/deeplab_v3_plus.yaml")
    cfg.freeze()

    model = DeepLabV3Plus(in_channels=3,
                          out_channels=21,
                          backbone="resnext50_32x4d",
                          aspp_cfg=cfg.MODEL.ASPP,
                          decoder_cfg=cfg.MODEL.DECODER,
                          output_stride=8)
    model = model.cuda()

    input = torch.rand(2, 3, 320, 320).cuda()
    feature = model(input, upsample_pred=False)
    print(feature.shape)
