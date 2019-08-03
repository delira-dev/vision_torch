from delira import get_backends

if "TORCH" in get_backends():
    import torch
    from torch.nn import functional as F
    from .utils import ConvNdTorch, PoolingNdTorch, NormNdTorch, DropoutNdTorch
    from .basic_networks import BaseSegmentationTorchNetwork

    class PSPModuleTorch(torch.nn.Module):
        def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6), n_dim=2):
            super().__init__()

            self.upsampling_mode = None
            if n_dim == 2:
                self.upsampling_mode = "bilinear"
            elif n_dim == 3:
                self.upsampling_mode = "trilinear"
            else:
                raise ValueError

            self._build_module(features, out_features, sizes, n_dim)

        def _build_module(self, features, out_features, sizes, n_dim):
            self.stages = torch.nn.ModuleList([self._make_stage(features, size, n_dim) for size in sizes])
            self.bottleneck = ConvNdTorch(n_dim, features*(len(sizes)+1), out_features, kernel_size=1)

        def _make_stage(self, features, size, n_dim):
            prior = PoolingNdTorch("AdaptiveAvg", 2, output_size=size)
            conv = ConvNdTorch(n_dim, features, features, kernel_size=1, bias=False)

            return torch.nn.Sequential(prior, conv)

        def forward(self, x):
            spatial_size = x.size()[2:]
            priors = [F.upsample(stage(x), size=spatial_size, mode=self.upsampling_mode) for stage in self.stages] + [x]

            return F.relu(self.bottleneck(torch.cat(priors, 1)))

    class PSPUpsampleTorch(torch.nn.Module):
        def __init__(self, in_channels, out_channels, n_dim, norm_type="Batch"):
            super().__init__()
            self.conv = torch.nn.Sequential(
                ConvNdTorch(n_dim, in_channels, out_channels, 3, padding=1),
                NormNdTorch(norm_type, n_dim, out_channels),
                torch.nn.PReLU()
            )

            self.upsampling_mode = None
            if n_dim == 2:
                self.upsampling_mode = "bilinear"
            elif n_dim == 3:
                self.upsampling_mode = "trilinear"
            else:
                raise ValueError

        def forward(self, x):
            spatial_size = [2*__size for __size in x.size()[2:]]

            return self.conv(F.upsample(x, size=spatial_size, mode=self.upsampling_mode))

    class PSPNet(BaseSegmentationTorchNetwork):
        def __init__(self, n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_feature_size=1024, backend="resnet18",
                     n_dim=2, norm_type="Batch", logsoftmax=True):

            super().__init__(n_classes, sizes, psp_size, deep_feature_size, backend,
                             n_dim, norm_type, logsoftmax)

        def _build_model(self, n_classes, sizes, psp_size, deep_feature_size, backend,
                         n_dim, norm_type, logsoftmax):
            from . import model_fns as backends

            # ToDo: remove classifier/last FC from backend net (maybe not necessary at all)
            self._backend = getattr(backends, backend + "_torch")

            self._psp = PSPModuleTorch(psp_size, 1024, sizes)
            self._drop_1 = DropoutNdTorch(n_dim, 0.3)

            self._up_1 = PSPUpsampleTorch(1024, 256, n_dim, norm_type)
            self._up_2 = PSPUpsampleTorch(256, 64, n_dim, norm_type)
            self._up_3 = PSPUpsampleTorch(64, 64, n_dim, norm_type)

            self._drop_2 = DropoutNdTorch(n_dim, 0.15)

            layers = [ConvNdTorch(n_dim, 64, n_classes, kernel_size=1)]

            if logsoftmax:
                layers.append(torch.nn.LogSoftmax())
            self._final = torch.nn.Sequential(
                *layers
            )

            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(deep_feature_size, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, n_classes)
            )

            self.adaptive_pool = PoolingNdTorch("AdaptiveMax", n_dim, 1)

        def forward(self, x):
            # ToDo: Change this to get the activations from the last 2 layers of backend (don't know how to use your hook)
            f, class_f = torch.rand(1, 2), torch.rand(1, 2)

            p = self._psp(f)
            p = self._drop_1(p)
            p = self._up_1(p)
            p = self._drop_2(p)
            p = self._up_2(p)
            p = self._drop_2(p)
            p = self._up_3(p)
            p = self.drop_2(p)

            auxiliary = self.adaptive_pool(class_f).view(-1, class_f.size(1))

            return self._final(p), self.classifier(auxiliary)