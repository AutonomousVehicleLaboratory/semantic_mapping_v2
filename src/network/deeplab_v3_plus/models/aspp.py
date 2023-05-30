from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nn.init import kaiming_normal
from core.nn.modules import Conv2d, DepthwiseSeparableConv2d


class AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module
    """

    def __init__(self,
                 in_channels,
                 out_channels=256,
                 atrous_channels=(256, 256, 256, 256),
                 atrous_kernel_size=(1, 3, 3, 3),
                 atrous_dilation=(1, 6, 12, 18),
                 dropout=0.5):
        """

        Args:
            in_channels (int): The number of input channels
            out_channels (int): The number of output channels
            atrous_channels (a list of int): The output channel of each atrous module
            atrous_kernel_size (a list of int): The kernel size of each atrous module
            atrous_dilation (a list of int): The dilation rate of each atrous module
            dropout (float): dropout probability

        Pooling may need crop size, but we do not implement here
        """
        super(AtrousSpatialPyramidPoolingModule, self).__init__()

        num_atours_layers = len(atrous_channels)
        assert num_atours_layers > 0
        assert num_atours_layers == len(atrous_kernel_size)
        assert num_atours_layers == len(atrous_dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Creates padding for each convolution layer so that their output size are the same
        atours_padding = []
        for d in atrous_dilation:
            padding = 0 if d == 1 else d
            atours_padding.append(padding)

        self.module_pyramid = nn.ModuleList()
        # Use simple conv for the first not dilated CNN
        conv = Conv2d(in_channels=in_channels, out_channels=atrous_channels[0],
                      kernel_size=atrous_kernel_size[0], dilation=atrous_dilation[0], padding=atours_padding[0],
                      bn=True, relu=True)
        self.module_pyramid.append(conv)
        # Use Depthwise Separable Conv2d for the rest of them
        for i in range(1, num_atours_layers):
            conv = DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=atrous_channels[i],
                                            kernel_size=atrous_kernel_size[i], dilation=atrous_dilation[i],
                                            padding=atours_padding[i],
                                            depthwise_bn=True, pointwise_bn=True,
                                            depthwise_relu=True, pointwise_relu=True)
            self.module_pyramid.append(conv)

        avg_pool_out_channel = 256
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Conv2d(in_channels, out_channels=avg_pool_out_channel, kernel_size=1, bn=True, relu=True),
        )

        num_cat_channel = np.sum(atrous_channels) + avg_pool_out_channel
        # Add bn and relu drops one percent of mIoU
        self.conv = Conv2d(in_channels=num_cat_channel, out_channels=out_channels, kernel_size=1, bn=True, relu=True)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, x):
        module_outputs = []
        for conv in self.module_pyramid:
            out = conv(x)
            module_outputs.append(out)

        # Add image pooling output
        output_size = module_outputs[0].shape[2:]
        out = self.global_avg_pool(x)
        out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=True)
        module_outputs.append(out)

        x = torch.cat(module_outputs, dim=1)
        x = self.conv(x)
        x = self.dropout(x)

        return x

    def init_weights(self):
        for module in self.module_pyramid:
            module.init_weights(kaiming_normal)
        for module in self.global_avg_pool:
            if isinstance(module, (Conv2d, DepthwiseSeparableConv2d)):
                module.init_weights(kaiming_normal)
        self.conv.init_weights(kaiming_normal)


if __name__ == '__main__':
    aspp = AtrousSpatialPyramidPoolingModule(in_channels=2048)
    input = torch.randn(5, 2048, 40, 40)
    output = aspp(input)
    print(output.shape)
