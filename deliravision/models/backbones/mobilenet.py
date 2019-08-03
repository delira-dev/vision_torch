from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from ..utils import ConvNdTorch, NormNdTorch
    from ..basic_networks import BaseClassificationTorchNetwork

    class ConvNormReLU(torch.nn.Sequential):
        def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                     groups=1, n_dim=2, norm_type="Batch"):
            padding = (kernel_size - 1) // 2

            super().__init__(
                ConvNdTorch(n_dim, in_planes, out_planes, kernel_size, stride,
                            padding, groups=groups, bias=False),
                NormNdTorch(norm_type, n_dim, out_planes),
                torch.nn.ReLU6(inplace=True)
            )


    class InvertedResidualTorch(torch.nn.Module):
        def __init__(self, inp, oup, stride, expand_ratio, n_dim=2,
                     norm_type="Batch"):
            super().__init__()
            self.stride = stride
            assert stride in [1, 2]

            hidden_dim = int(round(inp * expand_ratio))
            self.use_res_connect = self.stride == 1 and inp == oup

            layers = []
            if expand_ratio != 1:
                # pw
                layers.append(ConvNormReLU(inp, hidden_dim, kernel_size=1,
                                           n_dim=n_dim, norm_type=norm_type))
            layers.extend([
                # dw
                ConvNormReLU(hidden_dim, hidden_dim, stride=stride,
                             groups=hidden_dim, n_dim=n_dim,
                             norm_type=norm_type),
                # pw-linear
                ConvNdTorch(n_dim, hidden_dim, oup, 1, 1, 0, bias=False),
                NormNdTorch(norm_type, n_dim, oup)
            ])
            self.conv = torch.nn.Sequential(*layers)

        def forward(self, x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)


    class MobileNetV2Torch(BaseClassificationTorchNetwork):
        def __init__(self, num_classes=1000, width_mult=1.0, n_dim=2,
                     norm_type="Batch"):
            super().__init__(num_classes, width_mult, n_dim, norm_type)

        def _build_model(self, num_classes, width_mult, n_dim,
                         norm_type) -> None:

            block = InvertedResidualTorch
            input_channel = 32
            last_channel = 1280
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            # building first layer
            input_channel = int(input_channel * width_mult)
            self.last_channel = int(last_channel * max(1.0, width_mult))
            features = [ConvNormReLU(3, input_channel, stride=2, n_dim=n_dim,
                                     norm_type=norm_type)]

            # building inverted residual blocks
            for t, c, n, s in inverted_residual_setting:
                output_channel = int(c * width_mult)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride,
                                          expand_ratio=t, n_dim=n_dim,
                                          norm_type=norm_type))
                    input_channel = output_channel
            # building last several layers
            features.append(ConvNormReLU(input_channel, self.last_channel,
                                         kernel_size=1, n_dim=n_dim,
                                         norm_type=norm_type))
            # make it nn.Sequential
            self.features = torch.nn.Sequential(*features)

            # building classifier
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(self.last_channel, num_classes),
            )

            self.squeeze_dims = list(range(2, n_dim+2))

            # weight initialization
            for m in self.modules():
                if isinstance(m, ConvNdTorch):
                    torch.nn.init.kaiming_normal_(m.conv.weight, mode='fan_out')
                    if m.conv.bias is not None:
                        torch.nn.init.zeros_(m.conv.bias)
                elif isinstance(m,  NormNdTorch):
                    if hasattr(m.norm, "weight") and m.norm.weight is not None:
                        torch.nn.init.ones_(m.norm.weight)

                    if hasattr(m.norm, "bias") and m.norm.bias is not None:
                        torch.nn.init.zeros_(m.norm.bias)

                elif isinstance(m, torch.nn.Linear):
                    torch.nn.init.normal_(m.weight, 0, 0.01)
                    torch.nn.init.zeros_(m.bias)

        def forward(self, x) -> dict:
            x = self.features(x)
            x = x.mean(self.squeeze_dims)
            x = self.classifier(x)
            return {"pred": x}
