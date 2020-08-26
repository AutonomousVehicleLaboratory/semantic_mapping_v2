import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn import Conv2d, DepthwiseSeparableConv2d
from core.nn.init import kaiming_normal


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 low_level_in_channels,
                 low_level_out_channels,
                 refine_channels=(256, 256),
                 refine_kernel_size=(3, 3)):
        super(Decoder, self).__init__()

        num_refinement_layers = len(refine_channels)
        assert num_refinement_layers == len(refine_kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_level_in_channels = low_level_in_channels

        # CNN to reduce the dimension of lower level features
        self.low_level_conv = Conv2d(in_channels=low_level_in_channels, out_channels=low_level_out_channels,
                                     kernel_size=1, bn=True, relu=True)

        self.refine_layers = nn.ModuleList()
        in_channels = self.low_level_conv.out_channels + in_channels
        for i in range(num_refinement_layers):
            conv = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=refine_channels[i],
                                            kernel_size=refine_kernel_size[i],
                                            depthwise_bn=True, pointwise_bn=True,
                                            depthwise_relu=True, pointwise_relu=True)
            self.refine_layers.append(conv)
            in_channels = conv.out_channels
        # Output layer
        conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.refine_layers.append(conv)

        self.init_weights()

    def forward(self, feature, low_level_feature):
        low_level_feature = self.low_level_conv(low_level_feature)
        feature = F.interpolate(feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=True)
        feature = torch.cat([feature, low_level_feature], dim=1)
        for module in self.refine_layers:
            feature = module(feature)
        return feature

    def init_weights(self):
        self.low_level_conv.init_weights(kaiming_normal)
        for module in self.refine_layers:
            if isinstance(module, (Conv2d, DepthwiseSeparableConv2d,)):
                module.init_weights(kaiming_normal)
