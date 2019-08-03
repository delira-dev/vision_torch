from delira import get_backends as __get_backends

__all__ = []

if "TORCH" in __get_backends():

    from .segmentation import BaseSegmentationTorchNetwork
    from .classification import BaseClassificationTorchNetwork

    __all__ += [
        "BaseSegmentationTorchNetwork",
        "BaseClassificationTorchNetwork"
    ]