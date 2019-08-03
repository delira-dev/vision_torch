import torch
from functools import reduce
from operator import mul


class Generator(torch.nn.Module):
    """
    Very simple discriminator model
    """
    def __init__(self, latent_dim, img_shape):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_shape : tuple
            shape of generated images (including channels, excluding batch
            dimension)

        """
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(torch.nn.BatchNorm1d(out_feat, 0.8))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            torch.nn.Linear(1024, reduce(mul, img_shape)),
            torch.nn.Tanh()
        )
        self._img_shape = img_shape

    def forward(self, z):
        """
        Forwards a noise batch through the network

        Parameters
        ----------
        z : :class:`torch.Tensor`
            the noise batch

        Returns
        -------
        :class:`torch.Tensor`
            the generated images

        """
        img = self.model(z)
        img = img.view(img.size(0), *self._img_shape)
        return img


class Discriminator(torch.nn.Module):
    """
    A very simple discriminator network
    """
    def __init__(self, img_shape):
        """

        Parameters
        ----------
        img_shape : tuple
            shape of input images (including channels, excluding batch
            dimension)
        """
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(reduce(mul, img_shape), 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        """
        Forwards an image batch through the network

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the input images

        Returns
        -------
        :class:`torch.Tensor`
            discrimination result

        """
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity
