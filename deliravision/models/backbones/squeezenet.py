from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from ..utils import ConvNdTorch, PoolingNdTorch
    from ..basic_networks import BaseClassificationTorchNetwork

    class FireTorch(torch.nn.Module):

        def __init__(self, inplanes, squeeze_planes,
                     expand1x1_planes, expand3x3_planes, n_dim=2):
            super().__init__()
            self.inplanes = inplanes
            self.squeeze = ConvNdTorch(n_dim, inplanes, squeeze_planes,
                                       kernel_size=1)
            self.squeeze_activation = torch.nn.ReLU(inplace=True)
            self.expand1x1 = ConvNdTorch(n_dim, squeeze_planes, expand1x1_planes,
                                       kernel_size=1)
            self.expand1x1_activation = torch.nn.ReLU(inplace=True)
            self.expand3x3 = ConvNdTorch(n_dim, squeeze_planes, expand3x3_planes,
                                       kernel_size=3, padding=1)
            self.expand3x3_activation = torch.nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.squeeze_activation(self.squeeze(x))
            return torch.cat([
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1)


    class SqueezeNetTorch(BaseClassificationTorchNetwork):

        def __init__(self, version=1.0, num_classes=1000, in_channels=3,
                     n_dim=2, pool_type="Max", p_dropout=0.5):

            super().__init__(version, num_classes, in_channels, n_dim,
                             pool_type, p_dropout)

        def _build_model(self, version, num_classes, in_channels, n_dim,
                         pool_type, p_dropout) -> None:
            if version not in [1.0, 1.1]:
                raise ValueError("Unsupported SqueezeNet version {version}:"
                                 "1.0 or 1.1 expected".format(version=version))

            self.num_classes = num_classes
            if version == 1.0:
                self.features = torch.nn.Sequential(
                    ConvNdTorch(n_dim, in_channels, 96, kernel_size=7, stride=2),
                    torch.nn.ReLU(inplace=True),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(96, 16, 64, 64),
                    FireTorch(128, 16, 64, 64),
                    FireTorch(128, 32, 128, 128),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(256, 32, 128, 128),
                    FireTorch(256, 48, 192, 192),
                    FireTorch(384, 48, 192, 192),
                    FireTorch(384, 64, 256, 256),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(512, 64, 256, 256),
                )
            else:
                self.features = torch.nn.Sequential(
                    ConvNdTorch(n_dim, 3, 64, kernel_size=3, stride=2),
                    torch.nn.ReLU(inplace=True),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(64, 16, 64, 64),
                    FireTorch(128, 16, 64, 64),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(128, 32, 128, 128),
                    FireTorch(256, 32, 128, 128),
                    PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2,
                                   ceil_mode=True),
                    FireTorch(256, 48, 192, 192),
                    FireTorch(384, 48, 192, 192),
                    FireTorch(384, 64, 256, 256),
                    FireTorch(512, 64, 256, 256),
                )

            # Final convolution is initialized differently form the rest
            final_conv = ConvNdTorch(n_dim, 512, self.num_classes, kernel_size=1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=p_dropout),
                final_conv,
                torch.nn.ReLU(inplace=True),
                PoolingNdTorch("AdaptiveAvg", n_dim, 1)
            )

            for m in self.modules():
                if isinstance(m, ConvNdTorch):
                    if m is final_conv:
                        torch.nn.init.normal_(m.conv.weight, mean=0.0, std=0.01)
                    else:
                        torch.nn.init.kaiming_uniform_(m.conv.weight)
                    if m.conv.bias is not None:
                        torch.nn.init.constant_(m.conv.bias, 0)

        def forward(self, x) -> dict:
            x = self.features(x)
            x = self.classifier(x)
            return {"pred": x.view(x.size(0), self.num_classes)}
