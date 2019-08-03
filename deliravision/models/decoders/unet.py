from delira import get_backends

if "TORCH" in get_backends():
    import torch
    import torch.nn as nn
    from ..utils import ConvNdTorch, NormNdTorch, PoolingNdTorch

    class UNetTorch(nn.Module):
        def __init__(self, strides, channels,
                     merge_mode='add', interp_mode='nearest',
                     n_dim=2, norm_layer="Batch", **kwargs):
            """
            UNet decoder network
            .. note:: The U-Net decoder does not include the final convolution
            at the top layer

            Parameters
            ----------
            strides : iterable of int
                define stride of respective feature map with respect to the
                previous feature map (should contain one element less than
                `channels`)
            channels : iterable of int
                number of channels of each feature maps
            merge_mode : str
                Defines which mode should be used to merge feature maps of skip
                connection. `cat` concatenates feature maps, `add` uses
                elementwise addition
            interp_mode : str
                if `transpose` a transposed convolution is used for upsampling,
                otherwise it defines the methods used in torch.interpolate
            n_dim : int
                dimensionality of convolutions
            norm_layer : str
                defines type of normalization layer
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
            self._interp_mode = interp_mode
            self._merge_mode = merge_mode
            self._kwargs = kwargs
            self._build_model(n_dim, norm_layer)

        def _build_model(self, n_dim, norm_layer):
            """
            Build the model

            Parameters
            ----------
            n_dim : int
                dimensionality of convolution

            Returns
            -------
            """
            self.num_layers = 0
            for i, _ in enumerate(self._channels):
                # convolutions between upsampling process
                if self._merge_mode == 'cat':
                    channel1 = 2 * self._channels[i] if \
                        i < len(self._channels) - 1 else self._channels[i]
                else:
                    channel1 = self._channels[i]

                p = nn.Sequential(
                    ConvNdTorch(n_dim, channel1, self._channels[i],
                                kernel_size=3, stride=1, padding=1),
                    NormNdTorch(norm_layer, n_dim, self._channels[i]),
                    nn.ReLU(),
                    ConvNdTorch(n_dim, self._channels[i], self._channels[i],
                                kernel_size=3, stride=1, padding=1),
                    NormNdTorch(norm_layer, n_dim, self._channels[i]),
                    nn.ReLU(),)

                setattr(self, 'P{}'.format(i), p)

                # upsampling (top layer does not need upsampling)
                if i != 0:
                    if self._interp_mode == 'transpose':
                        up = ConvNdTorch(n_dim, self._channels[i],
                                         self._channels[i - 1],
                                         kernel_size=self._strides[i - 1],
                                         stride=self._strides[i - 1],
                                         transposed=True)
                    else:
                        up = nn.Sequential(
                            torch.nn.Upsample(mode=self._interp_mode,
                                              scale_factor=self._strides[i - 1],
                                              **self._kwargs),
                            ConvNdTorch(n_dim, self._channels[i],
                                        self._channels[i - 1], kernel_size=3,
                                        stride=1, padding=1),
                            nn.ReLU(),
                            NormNdTorch(norm_layer, n_dim, self._channels[i - 1]),)
                    setattr(self, 'P_up{}'.format(i), up)
                self.num_layers += 1

        def forward(self, inp_list):
            """
            Apply unet decoder to list of feature maps

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
                # get convolution
                conv = getattr(self, 'P{}'.format(self.num_layers - idx))

                # combine features from below
                if idx != 1:
                    if self._merge_mode == 'cat':
                        inp = torch.cat((inp, up), dim=1)
                    else:
                        inp = inp + up

                # compute out convolution
                out = conv(inp)
                out_list.append(out)

                # upsampling
                if idx != self.num_layers:
                    up_conv = getattr(self, 'P_up{}'.format(self.num_layers - idx))
                    up = up_conv(out)
            return out_list[::-1]
