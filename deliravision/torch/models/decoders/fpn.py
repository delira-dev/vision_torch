import torch
import torch.nn as nn
from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch


class FPNTorch(nn.Module):
    def __init__(self, strides, channels, out_channels,
                 interp_mode='nearest', n_dim=2, norm_type="Batch",
                 **kwargs):
        """
        FPN decoder network

        Parameters
        ----------
        strides : iterable of int
            define stride of respective feature map with respect to the
            previous feature map (should contain one element less than
            `channels`)
        channels : iterable of int
            number of channels of each feature maps
        out_channels : int
            number of output channels
        interp_mode : str
            if `transpose` a transposed convolution is used for upsampling,
            otherwise it defines the methods used in torch.interpolate
        n_dim : int
            dimensionality of convolutions
        kwargs
            additonal keyword arguments passed to interpolation function
        """
        super().__init__()
        if len(strides) + 1 != len(channels):
            raise ValueError("Strides must contain one element less than "
                             "channels.")
        assert len(channels) > 0

        self._strides = strides
        self._channels = channels
        self._out_channels = out_channels
        self._interp_mode = interp_mode
        self._kwargs = kwargs
        self._build_model(n_dim, norm_type)

    def _build_model(self, n_dim, norm_type):
        """
        Build the model

        Parameters
        ----------
        n_dim : int
            dimensionality of convolution

        Returns
        -------
        """
        # create model
        self.num_layers = 0
        for i, _ in enumerate(self._channels):
            # create convolution in skip connection
            p_lat = nn.Sequential(
                ConvNdTorch(n_dim, self._channels[i],
                            self._out_channels,
                            kernel_size=1, stride=1, padding=0),
                NormNdTorch(norm_type, n_dim, self._out_channels),
                nn.ReLU(inplace=True))
            setattr(self, 'P_lateral{}'.format(i), p_lat)

            # output convolution
            p_out = nn.Sequential(
                ConvNdTorch(n_dim, self._out_channels,
                            self._out_channels,
                            kernel_size=3, stride=1, padding=1),
                NormNdTorch(norm_type, n_dim, self._out_channels),
                nn.ReLU(inplace=True))
            setattr(self, 'P_out{}'.format(i), p_out)

            # upsampling (top layer does not need upsampling)
            if i != 0:
                if self._interp_mode == 'transpose':
                    up = ConvNdTorch(n_dim, self._out_channels,
                                     self._out_channels,
                                     kernel_size=self._strides[i - 1],
                                     stride=self._strides[i - 1],
                                     transposed=True)
                else:
                    up = torch.nn.Upsample(mode=self._interp_mode,
                                           scale_factor=self._strides[i - 1],
                                           **self._kwargs)
                setattr(self, 'P_up{}'.format(i), up)

            self.num_layers += 1

    def forward(self, inp_list):
        """
        Apply feature pyramid network to list of feature maps

        Parameters
        ----------
        inp_list : list
            list of feature maps

        Returns
        -------
        list
            list of output feature maps
        """
        out_list = []
        for idx, inp in enumerate(reversed(inp_list), 1):
            lateral_conv = getattr(
                self, 'P_lateral{}'.format(
                    self.num_layers - idx))
            out_conv = getattr(self, 'P_out{}'.format(self.num_layers - idx))

            # compute lateral connection
            lateral = lateral_conv(inp)

            # combine features from below
            if idx != 1:
                lateral = lateral + up

            # upsampling
            if idx != self.num_layers:
                up_conv = getattr(self, 'P_up{}'.format(self.num_layers - idx))
                up = up_conv(lateral)

            # compute output
            out_list.append(out_conv(lateral))
        return out_list[::-1]
