import torch
from ..utils import ConvNdTorch, PoolingNdTorch
from ..basic_networks import BaseClassificationTorchNetwork


class AlexNetTorch(BaseClassificationTorchNetwork):
    def __init__(self, num_classes=1000, in_channels=3, n_dim=2,
                 pool_type="Max"):
        super().__init__(num_classes, in_channels, n_dim, pool_type)

    def _build_model(self, num_classes, in_channels, n_dim,
                     pool_type) -> None:
        self.features = torch.nn.Sequential(
            ConvNdTorch(n_dim, in_channels, 64, kernel_size=11, stride=4,
                        padding=2),
            torch.nn.ReLU(inplace=True),
            PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2),
            ConvNdTorch(n_dim, 64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2),
            ConvNdTorch(n_dim, 192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            ConvNdTorch(n_dim, 384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            ConvNdTorch(n_dim, 256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            PoolingNdTorch(pool_type, n_dim, kernel_size=3, stride=2),
        )
        self.avgpool = PoolingNdTorch("AdaptiveAvg", n_dim, 6)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * pow(6, n_dim), 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x) -> dict:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return {"pred": x}
