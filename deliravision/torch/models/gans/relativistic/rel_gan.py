from deliravision.models.gans.relativistic.models import Discriminator, \
    Generator

from delira.models import AbstractPyTorchNetwork
import torch


class RelativisticGAN(AbstractPyTorchNetwork):
    """
        Class implementing generative adversarial networks with relative
        discriminators

        References
        ----------
        `Paper <https://arxiv.org/abs/1807.00734>`_

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

    def __init__(self, img_size, num_channels, latent_dim, rel_avg_gan=True,
                 generator_cls=Generator, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        img_size : int
            number of pixels per side of the image
        num_channels : int
            number of image channels to generate
        latent_dim : int
            size of the latent dimension
        rel_avg_gan : bool
            whether to apply relative discriminative loss functions or
            non-relative ones
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        """
        super().__init__()

        self.generator = generator_cls(img_size, num_channels, latent_dim)
        self.discriminator = discriminator_cls(num_channels, img_size)

        self._latent_dim = latent_dim
        self.rel_avg_gan = rel_avg_gan

    def forward(self, x, z=None):
        """
        Feeds a set of batches through all necessary networks

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch consisting of real images
        z : :class:`torch.Tensor`
            the noise batchn consisting of several noise vectors;
            will be sampled if not given

        Returns
        -------
        dict
            a dictionary containing all the (intermediate) results necessary
            for loss calculation and training

        """
        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        gen_imgs = self.generator(z)

        discr_real = self.discriminator(x)
        discr_fake = self.discriminator(gen_imgs)

        return {"gen_imgs": gen_imgs, "discr_real": discr_real,
                "discr_fake": discr_fake}

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

        if isinstance(model, torch.nn.DataParallel):
            rel_avg_gan = model.module.rel_avg_gan
        else:
            rel_avg_gan = model.rel_avg_gan

        metric_vals, loss_vals = {}, {}

        preds = model(data_dict["data"])

        if rel_avg_gan:
            loss_g = losses["adversarial"](
                preds["discr_fake"] -
                preds["discr_real"].mean(
                    0,
                    keepdim=True),
                True)
        else:
            loss_g = losses["adversarial"](
                preds["discr_fake"] - preds["discr_real"], True)

        loss_vals["generator"] = loss_g.item()

        optimizers["generator"].zero_grad()
        loss_g.backward(retain_graph=True)
        optimizers["generator"].step()

        if rel_avg_gan:
            discr_fake = losses["adversarial"](
                preds["discr_fake"] -
                preds["discr_real"].mean(
                    0,
                    keepdim=True),
                False)

            discr_real = losses["adversarial"](
                preds["discr_real"] -
                preds["discr_fake"].mean(
                    0,
                    keepdim=True),
                True)

        else:
            discr_fake = losses["adversarial"](
                preds["discr_fake"] - preds["discr_real"], False)
            discr_real = losses["adversarial"](
                preds["discr_real"] - preds["discr_fake"], True)

        loss_discr = (discr_real + discr_fake) / 2

        optimizers["discriminator"].zero_grad()
        loss_discr.backward()
        optimizers["discriminator"].step()

        loss_vals["discr_real"] = discr_real.item()
        loss_vals["discr_fake"] = discr_fake.item()
        loss_vals["discriminator"] = loss_discr.item()

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
        return {"data": torch.from_numpy(batch["data"]).to(torch.float
                                                           ).to(input_device)}
