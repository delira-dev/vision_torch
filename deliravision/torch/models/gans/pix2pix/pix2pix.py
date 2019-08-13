from deliravision.models.gans.pix2pix.models import Discriminator, \
    GeneratorUNet

from delira.models import AbstractPyTorchNetwork
import torch


class Pix2Pix(AbstractPyTorchNetwork):
    """
    Class implementing the
    Image-to-Image Translation with Conditional Adversarial Networks

    References
    ----------
    `Paper <https://arxiv.org/abs/1611.07004>`_

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

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 lambda_pixel: float = 100.,
                 generator_cls=GeneratorUNet, discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        in_channels : int
            number of channels per image of the source domain
        out_channels : int
            number of channels per image of the target domain
        lambda_pixel : float
            weighting factor for the pixelwise loss
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """

        super().__init__()

        self.generator = generator_cls(in_channels, out_channels)
        self.discriminator = discriminator_cls(in_channels, out_channels)
        self.lambda_pixel = lambda_pixel

    def forward(self, imgs_a, imgs_b):
        """
        Forwards one image batch per domain through the corresponding networks

        Parameters
        ----------
        imgs_a : :class:`torch.Tensor`
            images of the source domain
        imgs_b : :class:`torch.tensor`
            images of the target domain

        Returns
        -------
        dict
            dictionary containing all (intermediate) outputs for loss
            calculation and training

        """
        fake_b = self.generator(imgs_a)
        discr_fake = self.discriminator(fake_b, imgs_a)
        discr_real = self.discriminator(imgs_b, imgs_a)

        return {"fake_b": fake_b, "discr_fake": discr_fake,
                "discr_real": discr_real}

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
        if isinstance(model, torch.nn.Parallel):
            lambda_pixel = model.module.lambda_pixel
        else:
            lambda_pixel = model.lambda_pixel

        metric_vals, loss_vals = {}, {}

        preds = model(data_dict["data_a"], data_dict["data_b"])

        loss_adv = losses["adversarial"](preds["discr_fake"], True)
        loss_pixel = losses["pixelwise"](preds["fake_b"], data_dict["data_b"])

        loss_generator = loss_adv + lambda_pixel * loss_pixel

        loss_vals["gen_adversarial"] = loss_adv.item()
        loss_vals["gen_pixelwise"] = loss_pixel.item()
        loss_vals["generator"] = loss_generator.item()

        optimizers["generator"].zero_grad()
        loss_generator.backward(retain_graph=True)
        loss_generator.step()

        discr_real = losses["adversarial"](preds["discr_real"], True)
        discr_fake = losses["adversarial"](preds["discr_fake"], False)

        loss_discr = (discr_real + discr_fake) / 2

        loss_vals["discr_real"] = discr_real.item()
        loss_vals["discr_fake"] = discr_fake.item()
        loss_vals["discriminator"] = loss_discr.item()

        optimizers["discriminator"].zero_grad()
        loss_discr.backward()
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
        return {
            "data_a": torch.from_numpy(batch["data_a"]).to(torch.float).to(
                input_device),
            "data_b": torch.from_numpy(batch["data_b"]).to(torch.long).to(
                input_device
            )
        }
