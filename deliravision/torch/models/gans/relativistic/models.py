import torch


class Generator(torch.nn.Module):
    """
    A generative network
    """
    def __init__(self, img_size, num_channels, latent_dim):
        """

        Parameters
        ----------
        img_size : int
            number of pixels per side of the generated image
        num_channels : int
            number of channels to generate
        latent_dim : int
            size of the latent dimension

        """
        super().__init__()

        self.init_size = img_size // 4
        self.l1 = torch.nn.Linear(latent_dim, 128 * self.init_size ** 2)

        self.conv_blocks = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, z):
        """
        Feeds a batch of noise through the network and generates images from it

        Parameters
        ----------
        z : :class:`torch.Tensor`
            the noise batch

        Returns
        -------
        :class:`torch.Tensor`
            the resulting image batch

        """
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(torch.nn.Module):
    """
    A discriminative network
    """
    def __init__(self, num_channels, img_size):
        """

        Parameters
        ----------
        num_channels : int
            number of image channels
        img_size : int
            number of pixels per side of the input images

        """
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     torch.nn.LeakyReLU(0.2, inplace=True),
                     torch.nn.Dropout2d(0.25)]
            if bn:
                block.append(torch.nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = torch.nn.Sequential(
            *discriminator_block(num_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.model(torch.rand(1, num_channels, img_size, img_size)
                             ).size(2)
        self.adv_layer = torch.nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        """
        Feeds a batch of images through the network to infer their validity
        (whether they were generated or are real images)

        Parameters
        ----------
        img : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the image batch

        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
