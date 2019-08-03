import math
import torch
import torch.nn as nn
from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch
from .resnet import conv1x1, conv3x3


class SELayer(nn.Module):
    def __init__(self, n_dim, channel, reduction=16):
        """
        Squeeze and Excitation Layer
        https://arxiv.org/abs/1709.01507

        Parameters
        ----------
        n_dim : int
            dimensionality of convolution
        channel : int
            number of input channel
        reduction : int
            channel reduction factor
        """
        super(SELayer, self).__init__()
        self.pool = PoolingNdTorch('AdaptiveAvg', n_dim, 1)
        self.fc = nn.Sequential(
            ConvNdTorch(n_dim, channel, channel // reduction, kernel_size=1,
                        bias=False),
            nn.ReLU(inplace=True),
            ConvNdTorch(n_dim, channel // reduction, channel, kernel_size=1,
                        bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward input through layer

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            output
        """
        y = self.pool(x)
        y = self.fc(y)
        return x * y


class SEBasicBlockTorch(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer="Batch", n_dim=2, reduction=16):
        """
        Squeeze and Excitation Basic ResNet block

        Parameters
        ----------
        inplanes : int
            number of input channels
        planes : int
            number of intermediate channels
        stride : int or tuple
            stride of first convolution
        downsample : nn.Module
            downsampling in residual path
        norm_layer : str
            type of normalisation layer
        n_dim : int
            dimensionality of convolution
        reduction : int
            reduction for squeeze and excitation layer
        """
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, n_dim=n_dim)
        self.bn1 = NormNdTorch(norm_layer, n_dim, planes)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, n_dim=n_dim)
        self.bn2 = NormNdTorch(norm_layer, n_dim, planes)

        self.downsample = downsample
        self.stride = stride

        self.selayer = SELayer(n_dim, planes * self.expansion,
                               reduction=reduction)

    def forward(self, x):
        """
        Forward input through block

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            output
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.selayer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneckTorch(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer="Batch", n_dim=2, reduction=16):
        """
        Squeeze and Excitation Bottleneck ResNet block

        Parameters
        ----------
        inplanes : int
            number of input channels
        planes : int
            number of intermediate channels
        stride : int or tuple
            stride of first convolution
        downsample : nn.Module
            downsampling in residual path
        norm_layer : str
            type of normalisation layer
        n_dim : int
            dimensionality of convolution
        reduction : int
            reduction for squeeze and excitation layer
        """
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, n_dim=n_dim)
        self.bn1 = NormNdTorch(norm_layer, n_dim, planes)
        self.conv2 = conv3x3(planes, planes, stride, n_dim=n_dim)
        self.bn2 = NormNdTorch(norm_layer, n_dim, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, n_dim=n_dim)
        self.bn3 = NormNdTorch(norm_layer, n_dim, planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.selayer = SELayer(n_dim, planes * self.expansion,
                               reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward input through block

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            output
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.selayer(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneckXTorch(nn.Module):
    expansion = 4
    start_filts = 64

    def __init__(self, in_channels, channels, stride, cardinality,
                 width, n_dim, norm_layer, reduction=16):
        """
        Squeeze and Excitation ResNeXt Block

        Parameters
        ----------
        in_channels : int
            number of input channels
        stride : int
            stride of 3x3 convolution layer
        cardinality : int
            number of convolution groups
        width : int
            width of resnext block
        n_dim : int
            dimensionality of convolutions
        norm_layer : str
            type of normalization layer
        reduction : int
            reduction for se layer
        """
        super().__init__()
        out_channels = channels * self.expansion
        if cardinality == 1:
            rc = channels
        else:
            width_ratio = channels * (width / self.start_filts)
            rc = cardinality * math.floor(width_ratio)

        self.conv_reduce = ConvNdTorch(n_dim, in_channels, rc, kernel_size=1,
                                       stride=1, padding=0, bias=False)
        self.bn_reduce = NormNdTorch(norm_layer, n_dim, rc)
        self.relu = nn.ReLU(inplace=True)

        self.conv_conv = ConvNdTorch(n_dim, rc, rc, kernel_size=3,
                                     stride=stride, padding=1,
                                     groups=cardinality, bias=False)
        self.bn = NormNdTorch(norm_layer, n_dim, rc)

        self.conv_expand = ConvNdTorch(n_dim, rc, out_channels, kernel_size=1,
                                       stride=1, padding=0, bias=False)
        self.bn_expand = NormNdTorch(norm_layer, n_dim, out_channels)

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut.add_module(
                'shortcut_conv', ConvNdTorch(n_dim, in_channels, out_channels,
                                             kernel_size=1, stride=stride,
                                             padding=0, bias=False))
            self.shortcut.add_module(
                'shortcut_bn', NormNdTorch(norm_layer, n_dim, out_channels))

        self.selayer = SELayer(n_dim, out_channels, reduction=reduction)

    def forward(self, x):
        """
        Forward input through block

        Parameters
        ----------
        x : torch.Tensor
            input

        Returns
        -------
        torch.Tensor
            output
        """
        identity = x

        out = self.conv_reduce(x)
        out = self.relu(self.bn_reduce(out))

        out = self.conv_conv(out)
        out = self.relu(self.bn(out))

        out = self.conv_expand(out)
        out = self.bn_expand(out)

        out = self.selayer(out)

        res = self.shortcut(identity)
        return self.relu(res + out)
