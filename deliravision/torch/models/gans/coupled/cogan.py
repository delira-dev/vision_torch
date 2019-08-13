from delira.models import AbstractPyTorchNetwork
import torch

from deliravision.models.gans.coupled.models import CoupledDiscriminators, \
    CoupledGenerators


class CoupledGAN(AbstractPyTorchNetwork):
    """
    An implementation of of coupled generative adversarial networks, which are
    capable of generating images with the same content in different domains.

    References
    ----------
    `Paper <https://arxiv.org/abs/1606.07536>`_

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

    def __init__(self, img_size, in_channels, latent_dim,
                 generator_cls=CoupledGenerators,
                 discriminator_cls=CoupledDiscriminators):
        """

        Parameters
        ----------
        img_size : int
            image size
        in_channels : int
            number of input channels
        latent_dim : int
            size of the latent dimension
        generator_cls :
            class implementing the actual coupled generator topology
        discriminator_cls :
            class implementing the actual coupled discriminator topology

        """

        super().__init__()

        self.generator = generator_cls(img_size, latent_dim, in_channels)
        self.discriminator = discriminator_cls(in_channels, img_size)
        self._latent_dim = latent_dim

    def forward(self, imgs_a, imgs_b, noise=None):
        """

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            images of domain A
        imgs_b : :class:`torch.Tensor`
            images of domain B
        noise : :class:`torch.Tensor`
            noise vector for image generation (will be sampled from normal
            dirstibution if not given)

        Returns
        -------
        dict
            dictionary containing all (intermediate) outputs necessary for loss
            calculation and training

        """
        if noise is None:
            noise = torch.randn(imgs_a.size(0), self._latent_dim,
                                device=imgs_a.device, dtype=imgs_a.dtype)

        gen_imgs_a, gen_imgs_b = self.generator(noise)

        discr_fake_a, discr_fake_b = self.discriminator(gen_imgs_a,
                                                        gen_imgs_b)

        discr_real_a, discr_real_b = self.discriminator(imgs_a, imgs_b)

        return {"gen_imgs_a": gen_imgs_a, "gen_imgs_b": gen_imgs_b,
                "discr_fake_a": discr_fake_a, "discr_fake_b": discr_fake_b,
                "discr_real_a": discr_real_a, "discr_real_b": discr_real_b}

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

        preds = model(data_dict["data_a"], data_dict["data_b"])

        loss_gen_a = losses["adversarial"](preds["discr_fake_a"], True)
        loss_gen_b = losses["adversarial"](preds["discr_fake_b"], True)

        loss_vals["gen_a"] = loss_gen_a.item()
        loss_vals["gen_b"] = loss_gen_b.item()

        loss_generator = (loss_gen_a + loss_gen_b) / 2
        loss_vals["generator"] = loss_generator.item()

        optimizers["generator"].zero_grad()
        loss_generator.backward(retain_graph=True)
        optimizers["generator"].step()

        loss_discr_real_a = losses["adversarial"](preds["discr_real_a"], True)
        loss_discr_real_b = losses["adversarial"](preds["discr_real_b"], True)
        loss_discr_fake_a = losses["adversarial"](preds["discr_fake_a"], False)
        loss_discr_fake_b = losses["adversarial"](preds["discr_fake_b"], False)

        loss_discrimintaor = (loss_discr_real_a +
                              loss_discr_real_b +
                              loss_discr_fake_a +
                              loss_discr_fake_b) / 4

        loss_vals.update({
            "loss_discr_real_a": loss_discr_real_a.item(),
            "loss_discr_real_b": loss_discr_real_b.item(),
            "loss_discr_fake_a": loss_discr_fake_a.item(),
            "loss_discr_fake_b": loss_discr_fake_b.item(),
            "discriminator": loss_discrimintaor.item()
        })

        optimizers["discriminator"].zero_grad()
        loss_discrimintaor.backward()
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
        for k, v in batch.items():
            batch[k] = torch.from_numpy(v).to(torch.float).to(input_device)

        return batch
