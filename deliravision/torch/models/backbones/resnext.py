import math
import copy
import torch
import torch.nn as nn
from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch
from ..basic_networks import BaseClassificationTorchNetwork


class BottleneckTorch(nn.Module):
    expansion = 4
    start_filts = 64
    """
    RexNeXt bottleneck type C
    (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, channels, stride, cardinality,
                 width, n_dim, norm_layer):
        """

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
        norm_layer: str
            type of normalization layer
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

    def forward(self, x):
        """
        Forward input through network

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            output of bottleneck block
        """
        identity = x

        out = self.conv_reduce(x)
        out = self.relu(self.bn_reduce(out))

        out = self.conv_conv(out)
        out = self.relu(self.bn(out))

        out = self.conv_expand(out)
        out = self.bn_expand(out)

        res = self.shortcut(identity)
        return self.relu(res + out)


class StartConv(nn.Module):
    def __init__(self, n_dim, norm_layer, in_channels, start_filts, mode='7x7'):
        """
        Defines different sequences of start convolutions

        Parameters
        ----------
        n_dim : int
            dimensionality of convolutions
        norm_layer : str
            type of normlization layer
        in_channels : int
            number of input channels
        start_filts : int
            number of channels after first convolution
        mode : str
            either '7x7' for default configuration (7x7 conv) or 3x3 for
            three consecutive convolutions as proposed in
            https://arxiv.org/abs/1812.01187
        """
        super().__init__()
        self._in_channels = in_channels
        self._start_filts = start_filts
        self._mode = mode

        if mode == '7x7':
            self.convs = nn.Sequential(
                *[ConvNdTorch(n_dim, in_channels, self._start_filts,
                              kernel_size=7, stride=2, padding=3, bias=False),
                    NormNdTorch(norm_layer, n_dim, self._start_filts)]
            )
        elif mode == '3x3':
            self.convs = nn.Sequential(
                *[ConvNdTorch(n_dim, in_channels, self._start_filts,
                              kernel_size=3, stride=2, padding=1, bias=False),
                    NormNdTorch(norm_layer, n_dim, self._start_filts),
                    ConvNdTorch(n_dim, self._start_filts, self._start_filts,
                                kernel_size=3, stride=1, padding=1, bias=False),
                    NormNdTorch(norm_layer, n_dim, self._start_filts),
                    ConvNdTorch(n_dim, self._start_filts, self._start_filts,
                                kernel_size=3, stride=1, padding=1, bias=False),
                    NormNdTorch(norm_layer, n_dim, self._start_filts)
                  ]
            )
        else:
            raise ValueError('{} is not a supported mode!'.format(mode))

    def forward(self, input):
        return self.convs(input)


class ResNeXtTorch(BaseClassificationTorchNetwork):
    """
    ResNeXt model architecture
    """

    def __init__(self, block, layers, num_classes, in_channels, cardinality,
                 width=4, start_filts=64, start_mode='7x7',
                 n_dim=2, norm_layer='Batch'):
        """

        Parameters
        ----------
        block : nn.Module
            ResNeXt block used to build network
        layers : list of int
            defines how many blocks should be used in each stage
        num_classes : int
            number of classes
        in_channels : int
            number of input channels
        cardinality : int
            cardinality (number of groups)
        width : int
            width of resnext block
        start_filts : int
            number of start filter (number of channels after first conv)
        start_mode : str
            either '7x7' for default configuration (7x7 conv) or 3x3 for
            three consecutive convolutions as proposed in
            https://arxiv.org/abs/1812.01187
        n_dim : int
            dimensionality of convolutions
        norm_layer : str
            type of normalization
        """
        super().__init__(block, layers, num_classes, in_channels,
                         cardinality, width, start_filts, start_mode,
                         n_dim, norm_layer)

    def _build_model(self, block, layers, num_classes, in_channels,
                     cardinality, width, start_filts, start_mode,
                     n_dim, norm_layer) -> None:

        self._cardinality = cardinality
        self._width = width
        self._start_filts = start_filts
        self._num_classes = num_classes
        self._block = block
        self._block.start_filts = start_filts

        self._layers = layers
        self.inplanes = copy.copy(self._start_filts)

        self.conv1 = StartConv(n_dim, norm_layer, in_channels,
                               start_filts, start_mode)
        self.maxpool = PoolingNdTorch("Max", n_dim=n_dim, kernel_size=3,
                                      stride=2, padding=1)

        for idx, _layers in enumerate(layers):
            stride = 1 if idx == 0 else 2
            planes = self._start_filts * pow(2, idx)
            _local_layer = self._make_layer(block,
                                            _layers,
                                            self.inplanes,
                                            planes,
                                            norm_layer=norm_layer,
                                            n_dim=n_dim,
                                            pool_stride=stride)

            setattr(self, "C%d" % (idx + 1), _local_layer)
            self.inplanes = planes * block.expansion
        self._num_layers = len(layers)

        self.avgpool = PoolingNdTorch("AdaptiveAvg", n_dim, 1)
        self.fc = nn.Linear(self.inplanes, num_classes)

    def _make_layer(self, block, num_layers, in_channels, channels,
                    norm_layer, n_dim, pool_stride=2):
        """
        Stack multiple blocks

        Parameters
        ----------
        block : nn.Module
            block used to build the module
        in_channels : int
            number of input channels
        norm_layer : str
            type of normalization layer
        n_dim : int
            dimensionality of convolutions
        pool_stride : int
            pooling stride

        Returns
        -------
        nn.Module
            stacks blocks in a sequential module
        """
        module = []
        module.append(block(in_channels, channels, pool_stride,
                            self._cardinality, self._width,
                            n_dim, norm_layer))

        in_channels = channels * block.expansion

        for i in range(1, num_layers):
            module.append(block(in_channels, channels, 1,
                                self._cardinality, self._width,
                                n_dim, norm_layer))
        return nn.Sequential(*module)

    def forward(self, inp) -> dict:
        """
        Forward input through network

        Parameters
        ----------
        inp : torch.Tensor
            input to network

        Returns
        -------
        torch.Tensor
            network output
        """
        x = self.conv1(inp)
        x = self.maxpool(x)

        for i in range(self._num_layers):
            x = getattr(self, "C%d" % (i + 1))(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {"pred": x}
