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
                 width, n_dim, norm_layer, avg_down):
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
            if avg_down:
                self.shortcut.add_module(
                    'shortcut_avg',
                    PoolingNdTorch(
                        "Avg",
                        n_dim=n_dim,
                        kernel_size=stride,
                        stride=stride))
                self.shortcut.add_module(
                    'shortcut_conv', ConvNdTorch(
                        n_dim, in_channels, out_channels,
                        kernel_size=1, stride=1,
                        padding=0, bias=False))
            else:
                self.shortcut.add_module(
                    'shortcut_conv', ConvNdTorch(
                        n_dim, in_channels, out_channels,
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


class ResNeXtTorch(BaseClassificationTorchNetwork):
    """
    ResNeXt model architecture
    """

    def __init__(self, block, layers, num_classes, in_channels, cardinality,
                 width=4, start_filts=64, n_dim=2, norm_layer='Batch',
                 deep_start=False, avg_down=False, **kwargs):
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
                         cardinality, width, start_filts,
                         n_dim, norm_layer, deep_start, avg_down, **kwargs)

    def _build_model(self, block, layers, num_classes, in_channels,
                     cardinality, width, start_filts, n_dim, norm_layer,
                     deep_start, avg_down, **kwargs) -> None:

        self._cardinality = cardinality
        self._width = width
        self._start_filts = start_filts
        self._num_classes = num_classes
        self._block = block
        self._block.start_filts = start_filts

        self._layers = layers
        self.inplanes = copy.copy(self._start_filts)

        if not deep_start:
            self.conv1 = ConvNdTorch(n_dim, in_channels, self.inplanes,
                                     kernel_size=7, stride=2, padding=3,
                                     bias=False)
        else:
            self.conv1 = torch.nn.Sequential(
                ConvNdTorch(n_dim, in_channels, self.inplanes,
                            kernel_size=3, stride=2, padding=1,
                            bias=False),
                NormNdTorch(norm_layer, n_dim, self.inplanes),
                torch.nn.ReLU(inplace=True),
                ConvNdTorch(n_dim, self.inplanes, self.inplanes,
                            kernel_size=3, stride=1, padding=1,
                            bias=False),
                NormNdTorch(norm_layer, n_dim, self.inplanes),
                torch.nn.ReLU(inplace=True),
                ConvNdTorch(n_dim, self.inplanes, self.inplanes,
                            kernel_size=3, stride=1, padding=1,
                            bias=False),
            )
        self.bn1 = NormNdTorch(norm_layer, n_dim, self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
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
                                            avg_down=avg_down,
                                            pool_stride=stride,
                                            **kwargs,
                                            )

            setattr(self, "C%d" % (idx + 1), _local_layer)
            self.inplanes = planes * block.expansion
        self._num_layers = len(layers)

        self.avgpool = PoolingNdTorch("AdaptiveAvg", n_dim, 1)
        self.fc = nn.Linear(self.inplanes, num_classes)

    def _make_layer(self, block, num_layers, in_channels, channels,
                    norm_layer, n_dim, avg_down, pool_stride=2, **kwargs):
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
                            n_dim, norm_layer, avg_down,
                            **kwargs))

        in_channels = channels * block.expansion

        for i in range(1, num_layers):
            module.append(block(in_channels, channels, 1,
                                self._cardinality, self._width,
                                n_dim, norm_layer, avg_down,
                                **kwargs))
        return nn.Sequential(*module)

    def forward(self, x) -> dict:
        """
        Forward input through network

        Parameters
        ----------
        x : torch.Tensor
            input to network

        Returns
        -------
        torch.Tensor
            network output
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(self._num_layers):
            x = getattr(self, "C%d" % (i + 1))(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {"pred": x}
