
import copy
import torch
from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch
from ..basic_networks import BaseClassificationTorchNetwork


def conv3x3(in_planes, out_planes, stride=1, n_dim=2):
    """3x3 convolution with padding"""
    return ConvNdTorch(n_dim, in_planes, out_planes, kernel_size=3,
                       stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1, n_dim=2):
    """1x1 convolution"""
    return ConvNdTorch(n_dim, in_planes, out_planes, kernel_size=1,
                       stride=stride, bias=False)


class BasicBlockTorch(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer="Batch", n_dim=2):
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

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckTorch(torch.nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer="Batch", n_dim=2):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, n_dim=n_dim)
        self.bn1 = NormNdTorch(norm_layer, n_dim, planes)
        self.conv2 = conv3x3(planes, planes, stride, n_dim=n_dim)
        self.bn2 = NormNdTorch(norm_layer, n_dim, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, n_dim=n_dim)
        self.bn3 = NormNdTorch(norm_layer, n_dim, planes * self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetTorch(BaseClassificationTorchNetwork):
    def __init__(self, block, layers, num_classes=1000, in_channels=3,
                 zero_init_residual=False, norm_layer="Batch", n_dim=2,
                 start_filts=64, deep_start=False, avg_down=False):
        super().__init__(block, layers, num_classes, in_channels,
                         zero_init_residual, norm_layer, n_dim, start_filts,
                         deep_start, avg_down)

    def _build_model(self, block, layers, num_classes, in_channels,
                     zero_init_residual, norm_layer, n_dim,
                     start_filts, deep_start, avg_down) -> None:
        self.start_filts = start_filts
        self.inplanes = copy.copy(start_filts)

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

        num_layers = 0
        for idx, _layers in enumerate(layers):
            stride = 1 if idx == 0 else 2
            planes = min(self.start_filts * pow(2, idx), self.start_filts * 8)
            _local_layer = self._make_layer(block, planes, _layers,
                                            stride=stride,
                                            norm_layer=norm_layer,
                                            n_dim=n_dim,
                                            avg_down=avg_down)

            setattr(self, "layer%d" % (idx + 1), _local_layer)
            num_layers += 1

        self.num_layers = num_layers

        self.avgpool = PoolingNdTorch("AdaptiveAvg", n_dim, 1)
        self.fc = torch.nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, ConvNdTorch):
                torch.nn.init.kaiming_normal_(m.conv.weight,
                                              mode='fan_out',
                                              nonlinearity='relu')

            elif isinstance(m, NormNdTorch):
                if hasattr(m.norm, "weight") and m.norm.weight is not None:
                    torch.nn.init.constant_(m.norm.weight, 1)
                if hasattr(m.norm, "bias") and m.norm.bias is not None:
                    torch.nn.init.constant_(m.norm.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each
        # residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckTorch):
                    torch.nn.init.constant_(m.bn3.norm.weight, 0)
                elif isinstance(m, BasicBlockTorch):
                    torch.nn.init.constant_(m.bn2.norm.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer="Batch",
                    n_dim=2, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not avg_down:
                downsample = torch.nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride,
                            n_dim=n_dim),
                    NormNdTorch(norm_layer, n_dim, planes * block.expansion),
                )
            else:
                downsample = torch.nn.Sequential(
                    PoolingNdTorch("Avg", n_dim=n_dim, kernel_size=stride,
                                   stride=stride),
                    conv1x1(self.inplanes, planes * block.expansion, 1,
                            n_dim=n_dim),
                    NormNdTorch(norm_layer, n_dim, planes * block.expansion),
                )

        layers = [block(self.inplanes, planes, stride, downsample,
                        norm_layer, n_dim=n_dim)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                norm_layer=norm_layer, n_dim=n_dim))

        return torch.nn.Sequential(*layers)

    def forward(self, x) -> dict:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(self.num_layers):
            x = getattr(self, "layer%d" % (i + 1))(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return {"pred": x}
