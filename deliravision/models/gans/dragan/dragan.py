from deliravision.models.gans.dragan.models import Discriminator, Generator
from delira.models import AbstractPyTorchNetwork
import torch

from deliravision.models.gans.utils import weights_init_normal

class DRAGAN(AbstractPyTorchNetwork):
    """
    Implementation of Generative Adversarial Networks following the convergence
    and stability guidance of ``On Convergence and Stability of GANs``

    References
    ----------
    `Paper <https://arxiv.org/abs/1705.07215>`_

    Warnings
    --------
    This Network is designed for training only; if you want to predict from an
    already trained network, it might be best, to split this network into its
    parts (i. e. separating the discriminator from the generator). This will
    give a significant boost in inference speed and a significant decrease in
    memory consumption, since no memory is allocated for additional weights of
    the unused parts and no inference is done for them. If this whole network
    is used, inferences might be done multiple times per network, to obtain
    all necessary (intermediate) outputs for training.

    """
    def __init__(self, latent_dim, num_channels, img_size, lambda_gp=10.,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        num_channels : int
            number of channels for image generation and discrimination
        img_size : int
            number of pixels per image side
        lambda_gp : float
            weighting factor for gradient penalty
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """
        super().__init__()

        self.generator = generator_cls(img_size, latent_dim, num_channels)
        self.discriminator = discriminator_cls(num_channels, img_size)
        self._latent_dim = latent_dim
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.lambda_gp = lambda_gp

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None,
                alpha: torch.Tensor = None):
        """

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        noise : :class:`torch.Tensor`
            a noise batch for image generation, will be sampled from random
            distribution if not provided
        alpha : :class:`torch.Tensor`
            random weighting factor for distortion of :param:`x`; will be
            sampled from random distribution if not provided

        Returns
        -------
        dict
            a dictionary containing all (intermediate) results necessary for
            loss calculation and training

        """

        if noise is None:
            noise = torch.randn(x.size(0), self._latent_dim, device=x.device,
                                dtype=x.dtype)

        gen_imgs = self.generator(noise)

        discr_real = self.discriminator(x)
        discr_fake = self.discriminator(gen_imgs)

        if alpha is None:
            # Random weight term for interpolation
            alpha = torch.randn_like(x)

        interpolates = (alpha * x
                        + ((1 - alpha) * (x + 0.5 * x.std()
                                          * torch.randn_like(x))))
        interpolates.requires_grad_(True)

        discr_interpolates = self.discriminator(interpolates)

        return {"gen_imgs": gen_imgs, "discr_real": discr_real,
                "discr_fake": discr_fake, "interpolates": interpolates,
                "discr_interpolates": discr_interpolates}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses=None,
                metrics=None, fold=0, **kwargs):
        """
        Function which handles prediction from batch, logging, loss calculation
        and optimizer step

        Parameters
        ----------
        model : :class:`delira.models.AbstractPyTorchNetwork`
           model to forward data through
        data_dict : dict
           dictionary containing the data
        optimizers : dict
           dictionary containing all optimizers to perform parameter update
        losses : dict
           Functions or classes to calculate losses
        metrics : dict
           Functions or classes to calculate other metrics
        fold : int
           Current Fold in Crossvalidation (default: 0)
        kwargs : dict
           additional keyword arguments

        Returns
        -------
        dict
           Metric values (with same keys as input dict metrics); will always
           be empty here
        dict
           Loss values (with same keys as input dict losses)
        dict
           Arbitrary number of predictions
        """

        loss_vals, metric_vals = {}, {}

        if isinstance(model, torch.nn.DataParallel):
            lambda_gp = model.module.lambda_gp
        else:
            lambda_gp = model.lambda_gp

        preds = model(data_dict["data"])

        gen_loss = losses["adversarial"](preds["discr_fake"], True)
        loss_vals["generator"] = gen_loss.item()

        optimizers["generator"].zero_grad()
        gen_loss.backward(retain_graph=True)
        optimizers["generator"].step()

        real_loss = losses["adversarial"](preds["discr_real"], True)
        fake_loss = losses["adversarial"](preds["discr_fake"], False)

        loss_vals["discr_real"] = real_loss.item()
        loss_vals["discr_fake"] = fake_loss.item()

        gradient_penalty = losses["gradient_penalty"](
            preds["discr_interpolates"], preds["interpolates"])

        discr_loss = (fake_loss + real_loss) / 2 + gradient_penalty * lambda_gp
        loss_vals["discriminator"] = discr_loss.item()

        optimizers["discriminator"].zero_grad()
        discr_loss.backward()
        optimizers["discriminator"].step()

        # zero gradients again just to make sure, gradients aren't carried to
        # next iteration (won't affect training since gradients are zeroed
        # before every backprop step, but would result in way higher memory
        # consumption)
        for k, v in optimizers.items():
            v.zero_grad()

        return metric_vals, loss_vals, {k: v.detach()
                                        for k, v in preds.items()}

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        """
        Helper Function to prepare Network Inputs and Labels (convert them
        to correct type and shape and push them to correct devices)
        Parameters
        ----------
        batch : dict
            dictionary containing all the data
        input_device : torch.device
            device for network inputs
        output_device : torch.device
            device for network outputs
        Returns
        -------
        dict
            dictionary containing data in correct type and shape and on
            correct device
        """
        return {"data":
                    torch.from_numpy(batch["data"]
                                     ).to(torch.float).to(input_device)}