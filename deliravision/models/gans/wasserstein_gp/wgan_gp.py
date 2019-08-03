from deliravision.models.gans.wasserstein_gp.models import Discriminator, \
    Generator

from delira.models import AbstractPyTorchNetwork

import torch


class WassersteinGradientPenaltyGAN(AbstractPyTorchNetwork):
    """
    Class implementing Improved Training of Wasserstein GANs

    References
    ----------
    `Paper <https://arxiv.org/abs/1704.00028>`_

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
    def __init__(self, img_shape, latent_dim, gen_update_freq=5, lambda_gp=10.,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_shape : tuple
            the shape of the real/generated images
        latent_dim : int
            size of the latent dimension
        lambda_gp : float
            the impact weight of the gradient penalty loss term w.r.t
            the whole loss
        gen_update_freq : int
            number of discriminator update steps to do per generator update
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """

        super().__init__()

        self.generator = generator_cls(img_shape, latent_dim)
        self.discriminator = discriminator_cls(img_shape)

        self._latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self._update_gen_freq = gen_update_freq
        self._update_gen_ctr = 0

    def forward(self, x, z=None, alpha=None):
        """
        Feeds a set of batches through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        z : :class:`torch.Tensor`
            the noise batch, will be sampled from normal distribution if not
            given
        alpha : :class:`torch.Tensor`
            a random value for interpolation between real and fake values;
            will be sampled if not given

        Returns
        -------
        dict
            a dictionary containing all relevant (intermediate) outputs
            necessary for loss calculation and training obtained by the
            different subnets

        """
        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        if alpha is None:
            alpha = torch.rand(x.size(0), 1, 1, 1, device=x.device,
                               dtype=x.dtype)

        gen_imgs = self.generator(z)

        interpolates = alpha * x + (1 - alpha) * gen_imgs
        interpolates.requires_grad_(True)
        discr_interpolates = self.discriminator(interpolates)
        discr_fake = self.discriminator(gen_imgs)
        discr_real = self.discriminator(x)
        return {"gen_imgs": gen_imgs, "discr_fake": discr_fake,
                "discr_real": discr_real, "interpolates": interpolates,
                "discr_interpolates": discr_interpolates}

    @property
    def update_gen(self):
        """
        A property whether to update the generator in the current iteration

        Returns
        -------
        bool
            whether to update the generator

        """
        try:
            if self._update_gen_ctr == 0:
                return True
            return False
        # incrementing the counter will always be done because the finally
        # block is always executed a try-except block is exitted - even after
        # a return statement
        finally:
            self._update_gen_ctr = ((self._update_gen_ctr + 1)
                                    % self._update_gen_freq)

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

        metric_vals, loss_vals = {}, {}

        if isinstance(model, torch.nn.DataParallel):
            attr_module = model.module
        else:
            attr_module = model

        preds = model(data_dict["data"])

        update_gen = attr_module.update_gen

        loss_adv = preds["discr_fake"].mean() - preds["discr_real"].mean()
        loss_gp = losses["gradient_penalty"](preds["discr_interpolates"],
                                             preds["interpolates"])
        loss_vals["discr_adversarial"] = loss_adv.item()
        loss_vals["discr_gradient_penalty"] = loss_gp.item()

        loss_d = loss_adv + attr_module.lambda_gp * loss_gp
        loss_vals["discriminator"] = loss_d.item()

        optimizers["discriminator"].zero_grad()
        loss_d.backward(retain_graph=update_gen)
        optimizers["discriminator"].step()

        loss_gen = -preds["discr_fake"].mean()
        loss_vals["generator"] = loss_gen.item()

        if update_gen:
            optimizers["generator"].zero_grad()
            loss_gen.backward()
            optimizers["generator"].step()

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
        return {
            "data": torch.from_numpy(batch["data"]).to(
                torch.float).to(input_device)}
