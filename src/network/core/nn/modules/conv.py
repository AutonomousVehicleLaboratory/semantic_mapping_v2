import numbers
import numpy as np
import torch.nn as nn


def padding2d_same_mode(kernel_size, stride, dilation):
    """
    Create a module that zero-pad the tensor with SAME padding mode (match to the TensorFlow implementation)

    Args:
        kernel_size: (height, width) or a int
        stride: (height, width) or a int
        dilation: (height, width) or a int

    Returns:
        The padding module
    Reference
        https://www.tensorflow.org/api_docs/python/tf/nn/convolution
        https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
    """
    if isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, numbers.Number):
        stride = (stride, stride)
    if isinstance(dilation, numbers.Number):
        dilation = (dilation, dilation)
    assert len(kernel_size) == 2, "kernel_size must be an int or a pair of int."
    assert len(stride) == 2, "stride must be an int or a pair of int."
    assert len(dilation) == 2, "dilation must be an int or a pair of int."

    kernel_size = np.array(kernel_size)
    stride = np.array(stride)
    dilation = np.array(dilation)

    pad_total = dilation * kernel_size - dilation + 1 - stride
    pad_total = np.clip(pad_total, 0, None)  # If pad_total < 0, we don't pad it.

    pad_top, pad_left = pad_total // 2
    pad_bottom = pad_total[0] - pad_top
    pad_right = pad_total[1] - pad_left
    return nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0)


class Conv1d(nn.Module):
    pass


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation) over an input signal
    composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (nn.Module): relu activation module
    """

    def __init__(self, in_channels, out_channels, kernel_size, bn=False, bn_momentum=0.1, relu=False,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode="zeros"):
        """

        Args:
            padding: support "zeros" or "same"
        """
        super(Conv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if padding_mode == "same":
            self.padding2d = padding2d_same_mode(kernel_size, stride, dilation)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not bn), stride=stride,
                                  padding=0, dilation=dilation, groups=groups, padding_mode="zeros")
        else:
            self.padding2d = None
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=(not bn), stride=stride,
                                  padding=padding, dilation=dilation, groups=groups, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        if self.padding2d is not None:
            x = self.padding2d(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def init_weights(self, init_fn=None):
        """
        Note we skipped the initialization of batch normalization because its default value (gamma=1 and beta=0) is
         good enough)

        Args:
            init_fn: Initialization function
        """
        if init_fn is not None:
            init_fn(self.conv)


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise 2D Separable convolution network: A spatial convolution performed independently over each channel,
    followed by a point-wise convolution (i.e. 1x1 convolution)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, padding_mode="zeros",
                 depthwise_bn=False, pointwise_bn=False, bn_momentum=0.1,
                 depthwise_relu=False, pointwise_relu=False):
        """

        Args:
            *same parameters as Conv2d
            depthwise_bn (bool): True if the depthwise_cnn should go through a Batch Normalization layer.
            pointwise_bn (bool): True if the pointwise_cnn should go through a Batch Normalization layer.
            depthwise_relu (bool): True if the depthwise_cnn should go through a relu function
            pointwise_relu (bool): True if the pointwise_cnn should go through a relu function
        """
        super(DepthwiseSeparableConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Note that
        # * depthwise cnn requires 'group = in_channels'
        # * All the pointwise cnn has batch normalization as stated in Xception paper (Figure 5)
        # * Point-wise CNN does not add stride or padding
        self.depthwise_cnn = Conv2d(in_channels, in_channels, kernel_size,
                                    bn=depthwise_bn, bn_momentum=bn_momentum, relu=depthwise_relu,
                                    stride=stride, padding=padding, dilation=dilation,
                                    groups=in_channels, padding_mode=padding_mode)
        self.pointwise_cnn = Conv2d(in_channels, out_channels, kernel_size=1,
                                    bn=pointwise_bn, bn_momentum=bn_momentum, relu=pointwise_relu)

    def forward(self, x):
        x = self.depthwise_cnn(x)
        x = self.pointwise_cnn(x)
        return x

    def init_weights(self, init_fn):
        self.depthwise_cnn.init_weights(init_fn)
        self.pointwise_cnn.init_weights(init_fn)
