# An implementation of Modified Aligned Xception
# Based on TensorFlow version: https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py
import torch.nn as nn

from core.nn import Conv2d, DepthwiseSeparableConv2d
from core.nn.init import kaiming_normal


class XceptionBlock(nn.Module):
    """
    An XceptionBlock is a network architecture that contains k (k>0) number of depth-wise separable CNNs (residual
    layers) and a shortcut layer which connects the input to the output.

    The output of the Block is the sum of 'residual' and 'shortcut', where 'residual' is the feature computed by k
    depth-wise separable CNN and 'shortcut' is the feature computed by 1x1 CNN with or without striding. In some
    cases, the 'shortcut' path could be a simple identity function or none (no shortcut).

    Modified from xception_module() in TensorFlow
    """

    def __init__(self,
                 in_channels,
                 residual_channel_list=(),
                 residual_kernel_size=(),
                 residual_stride=(),
                 residual_dilation=(),
                 skip_connection_type=None,
                 skip_connection_channel=0,
                 skip_connection_kernel=0,
                 skip_connection_stride=0,
                 entry_relu=True,
                 return_residual_features=False,
                 add_residual_padding=False):
        """

        Args:
            in_channels: Number of channels in input
            residual_channel_list (a list of int): The output channel of each depth-wise separable CNN in residual path
            residual_kernel_size (a list of int): The kernel size of each depth-wise separable CNN in residual path
            residual_stride: The stride size of each depth-wise separable CNN in residual path
            residual_dilation: The dilation rate of each depth-wise separable CNN in residual path
            skip_connection_type: The type of skip connection. It can be 'conv', 'sum' or None.
            skip_connection_channel: Number of channel for skip connection if it is 'conv'
            skip_connection_kernel: The kernel size if skip connection is 'conv'
            skip_connection_stride: The stride size if skip connection is 'conv'
            entry_relu: True if the input need to go through a ReLU layer, False otherwise.
                In the Deformable Convolution Paper, the relu function is put at the end of the inception
                block, but here because we may want to extract the low level feature without capping it
                with nonlinearity, we move the relu function to the front.
            return_residual_features: Create a channel for User to access to the feature before the last residual
                model. This enables users to access to low level features
            add_residual_padding: True if we add a padding before the last residual module. Usually this is True
                when the skip layer has kernel_size = 1 and stride = 2.
        """
        super(XceptionBlock, self).__init__()

        num_residual_conv = len(residual_channel_list)

        # ===== Sanity Check =====
        assert num_residual_conv > 0, "The number of separable convolution must be greater than zero!"
        assert num_residual_conv == len(residual_kernel_size)
        assert num_residual_conv == len(residual_stride)

        if skip_connection_type is not None:
            assert skip_connection_type in ["conv", "sum"], \
                "Invalid skip connection type: {}!".format(skip_connection_type)

        if skip_connection_type == "conv":
            assert skip_connection_channel == residual_channel_list[-1]
        # ========================

        self.in_channels = in_channels
        # Note that inplace has to be False because skip layer needs the original tensor
        self.entry_relu = nn.ReLU(inplace=False) if entry_relu else None
        self.return_residual_features = return_residual_features

        # Separate the residual modules into two groups: the last separable CONV is one group and the rest of them
        # are another. The purpose of the last separable CONV is maxpooling so it has to be treated differently.
        # Note:
        # * SeparableConv2d does not have ReLU after depthwise CNN (from official TensorFlow implementation) even
        # though the paper claims it does
        self.residual_group1 = nn.ModuleList()
        res_in_channel = in_channels
        for i in range(num_residual_conv - 1):
            conv = DepthwiseSeparableConv2d(in_channels=res_in_channel,
                                            out_channels=residual_channel_list[i],
                                            kernel_size=residual_kernel_size[i],
                                            stride=residual_stride[i],
                                            dilation=residual_dilation[i],
                                            padding_mode="same",
                                            depthwise_bn=True,
                                            pointwise_bn=True)
            self.residual_group1.append(conv)
            # We manually add relu into the block because if user wants to get the low level features, the low level
            # feature is the output of the residual_group1 without going through ReLU.
            # inplace has to be false because user may need the low level features.
            self.residual_group1.append(nn.ReLU(inplace=False))
            res_in_channel = conv.out_channels

        self.residual_group2 = nn.ModuleList()
        if add_residual_padding:
            self.residual_group2.append(nn.ConstantPad2d((0, 1, 0, 1), value=0))
        # We do not add pointwise relu in the last CNN
        conv = DepthwiseSeparableConv2d(in_channels=res_in_channel,
                                        out_channels=residual_channel_list[-1],
                                        kernel_size=residual_kernel_size[-1],
                                        stride=residual_stride[-1],
                                        dilation=residual_dilation[-1],
                                        padding_mode="same",
                                        depthwise_bn=True,
                                        pointwise_bn=True)
        self.residual_group2.append(conv)

        # Skip path
        self.skip_connection_type = skip_connection_type
        self.skip_connection = None
        if skip_connection_type == "conv":
            self.skip_connection = Conv2d(in_channels=in_channels,
                                          out_channels=skip_connection_channel,
                                          kernel_size=skip_connection_kernel,
                                          stride=skip_connection_stride,
                                          padding_mode="same",
                                          bn=True)

        self.out_channels = residual_channel_list[-1]

        self.init_weights()

    def forward(self, x):
        low_level_feature = None
        # Residual path
        residual = x if self.entry_relu is None else self.entry_relu(x)
        for module in self.residual_group1:
            if isinstance(module, nn.ReLU):
                low_level_feature = residual
            residual = module(residual)
        for module in self.residual_group2:
            residual = module(residual)

        # Skip connection path
        if self.skip_connection_type == "conv":
            shortcut = self.skip_connection(x)
        elif self.skip_connection_type == "sum":
            shortcut = x
        else:
            shortcut = 0

        output = residual + shortcut
        if self.return_residual_features:
            return output, low_level_feature
        else:
            return output

    def init_weights(self):
        for module in self.residual_group1:
            if isinstance(module, (DepthwiseSeparableConv2d,)):
                module.init_weights(kaiming_normal)

        for module in self.residual_group2:
            if isinstance(module, (DepthwiseSeparableConv2d,)):
                module.init_weights(kaiming_normal)

        if isinstance(self.skip_connection, Conv2d):
            self.skip_connection.init_weights(kaiming_normal)


class Xception65(nn.Module):
    """
    An Xception 65 module

    Its architecture is the following
    Entry flow:
        First go through two layers of normal CNN. Then go through a series of XceptionBlocks.
    Middle flow:
        A concatenation of XceptionBlocks
    Exit flow:
        A series of XceptionBlocks + Depthwise separable CNNs
        Fully connected layer is not implement inside this module. Users are welcome to add it outside the module.
    """

    def __init__(self,
                 in_channels,
                 return_low_level_feature=False):
        """

        Args:
            in_channels (int)
            return_low_level_feature (bool): True if return the low level features
        """
        super(Xception65, self).__init__()

        self.return_low_level_feature = return_low_level_feature
        self.low_level_feature_channels = 256  # Record the low level feature right here.
        self.in_channel = in_channels
        self.out_channel = 2048

        # Build Entry flow CNNs
        self.entry_flow_modules = nn.ModuleList()
        conv = Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, bn=True, relu=True)
        self.entry_flow_modules.append(conv)

        conv = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bn=True, relu=True, padding_mode="same")
        self.entry_flow_modules.append(conv)
        in_channels = conv.out_channels

        # Entry flow blocks
        block = XceptionBlock(in_channels=in_channels,
                              residual_channel_list=(128, 128, 128),
                              residual_kernel_size=(3, 3, 3),
                              residual_stride=(1, 1, 2),
                              residual_dilation=(1, 1, 1),
                              skip_connection_type="conv",
                              skip_connection_channel=128,
                              skip_connection_kernel=1,
                              skip_connection_stride=2,
                              add_residual_padding=True)
        self.entry_flow_modules.append(block)
        in_channels = block.out_channels

        block = XceptionBlock(in_channels=in_channels,
                              residual_channel_list=(256, 256, 256),
                              residual_kernel_size=(3, 3, 3),
                              residual_stride=(1, 1, 2),
                              residual_dilation=(1, 1, 1),
                              skip_connection_type="conv",
                              skip_connection_channel=256,
                              skip_connection_kernel=1,
                              skip_connection_stride=2,
                              return_residual_features=return_low_level_feature,
                              add_residual_padding=True)
        self.entry_flow_modules.append(block)
        in_channels = block.out_channels

        block = XceptionBlock(in_channels=in_channels,
                              residual_channel_list=(728, 728, 728),
                              residual_kernel_size=(3, 3, 3),
                              residual_stride=(1, 1, 2),
                              residual_dilation=(1, 1, 1),
                              skip_connection_type="conv",
                              skip_connection_channel=728,
                              skip_connection_kernel=1,
                              skip_connection_stride=2,
                              add_residual_padding=True)
        self.entry_flow_modules.append(block)
        in_channels = block.out_channels

        # Middle flow blocks
        self.middle_flow_modules = nn.ModuleList()
        num_blocks = 16
        for i in range(num_blocks):
            block = XceptionBlock(in_channels=in_channels,
                                  residual_channel_list=(728, 728, 728),
                                  residual_kernel_size=(3, 3, 3),
                                  residual_stride=(1, 1, 1),
                                  residual_dilation=(1, 1, 1),
                                  skip_connection_type="sum")
            self.middle_flow_modules.append(block)
            in_channels = block.out_channels

        # Exit flow blocks
        self.exit_flow_modules = nn.ModuleList()
        block = XceptionBlock(in_channels=in_channels,
                              residual_channel_list=(728, 1024, 1024),
                              residual_kernel_size=(3, 3, 3),
                              residual_stride=(1, 1, 1),
                              residual_dilation=(1, 1, 1),
                              skip_connection_type="conv",
                              skip_connection_channel=1024,
                              skip_connection_kernel=1,
                              skip_connection_stride=1)
        self.exit_flow_modules.append(block)
        in_channels = block.out_channels

        conv_channels = (1536, 1536, 2048)
        kernel_size = (3, 3, 3)
        for i in range(len(conv_channels)):
            conv = DepthwiseSeparableConv2d(in_channels=in_channels,
                                            out_channels=conv_channels[i],
                                            kernel_size=kernel_size[i],
                                            stride=1,
                                            dilation=1,
                                            padding_mode="same",
                                            depthwise_bn=True,
                                            depthwise_relu=True,
                                            pointwise_bn=True,
                                            pointwise_relu=True)
            self.exit_flow_modules.append(conv)
            in_channels = conv.out_channels

        self.out_channel = in_channels

        self.init_weights()

    def forward(self, x):
        low_level_features = None

        for module in self.entry_flow_modules:
            x = module(x)
            if isinstance(x, tuple):
                x, low_level_features = x

        for module in self.middle_flow_modules:
            x = module(x)
        for module in self.exit_flow_modules:
            x = module(x)

        if self.return_low_level_feature:
            return x, low_level_features
        else:
            return x

    def init_weights(self):
        for module in self.entry_flow_modules:
            if isinstance(module, (Conv2d, DepthwiseSeparableConv2d)):
                module.init_weights(kaiming_normal)
            else:
                module.init_weights()

        for module in self.entry_flow_modules:
            module.init_weights()

        for module in self.exit_flow_modules:
            if isinstance(module, (Conv2d, DepthwiseSeparableConv2d)):
                module.init_weights(kaiming_normal)
            else:
                module.init_weights()


if __name__ == '__main__':
    def test_Xception():
        import torch

        # input = torch.rand((5, 3, 515, 515)).cuda()
        input = torch.rand((15, 3, 299, 299)).cuda()
        xception = Xception65(in_channels=3, return_low_level_feature=False)
        xception = xception.cuda()
        print(xception)

        feature = xception(input)
        print(feature.shape)

        # feature, low_level_feature = xception(input)
        # print(feature.shape, low_level_feature.shape)


    test_Xception()
