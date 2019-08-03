from delira import get_backends

if "TORCH" in get_backends():
    from .nd_wrapper_torch import ConvWrapper as ConvNdTorch, \
        NormWrapper as NormNdTorch, PoolingWrapper as PoolingNdTorch, DropoutWrapper as DropoutNdTorch
