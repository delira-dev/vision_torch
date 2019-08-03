from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch
    from ..basic_networks import BaseClassificationTorchNetwork

    class VGGTorch(BaseClassificationTorchNetwork):

        def __init__(self, feature_cfg, num_classes=1000, in_channels=3,
                     init_weights=True, n_dim=2, norm_type="Batch",
                     pool_type="Max"):
            super().__init__(feature_cfg, num_classes, in_channels,
                             init_weights, n_dim, norm_type, pool_type)

        def _build_model(self, feature_cfg, num_classes, in_channels,
                         init_weights, n_dim, norm_type, pool_type) -> None:
            self.features = self.make_layers(feature_cfg, norm_type=norm_type,
                                             n_dim=n_dim,
                                             in_channels=in_channels,
                                             pool_type=pool_type)

            self.avgpool = PoolingNdTorch("AdaptiveAvg", n_dim, 7)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512 * pow(7, n_dim), 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, num_classes),
            )
            if init_weights:
                self._initialize_weights()

        def forward(self, x) -> dict:
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return {"pred": x}

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, ConvNdTorch):
                    torch.nn.init.kaiming_normal_(m.conv.weight,
                                                  mode='fan_out',
                                                  nonlinearity='relu')
                    if m.conv.bias is not None:
                        torch.nn.init.constant_(m.conv.bias, 0)
                elif isinstance(m, NormNdTorch):
                    if hasattr(m.norm, "weight") and m.norm.weight is not None:
                        torch.nn.init.constant_(m.norm.weight, 1)

                    if hasattr(m.norm, "bias") and m.norm.bias is not None:
                        torch.nn.init.constant_(m.norm.bias, 0)

                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                    torch.nn.init.constant_(m.bias, 0)

        @staticmethod
        def make_layers(cfg, norm_type=None, n_dim=2, in_channels=3,
                        pool_type="Max"):
            layers = []

            for v in cfg:
                if v == 'P':
                    layers += [PoolingNdTorch(pool_type, n_dim, kernel_size=2,
                                              stride=2)]
                else:
                    _layers = [ConvNdTorch(n_dim, in_channels, v, kernel_size=3,
                                           padding=1)]
                    if norm_type is not None:
                        _layers.append(NormNdTorch(norm_type, n_dim, v))

                    _layers.append(torch.nn.ReLU(inplace=True))
                    layers += _layers
                    in_channels = v

            return torch.nn.Sequential(*layers)
