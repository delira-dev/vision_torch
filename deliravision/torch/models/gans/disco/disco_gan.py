import torch

from deliravision.models.gans.disco.models import Discriminator, GeneratorUNet
from delira.models import AbstractPyTorchNetwork


class DiscoGAN(AbstractPyTorchNetwork):
    """
    Implementation of Generative Adversarial Networks for discovery of
    cross-domain relations in image synthesis with exchangeable generator and
    discriminator classes

    References
    ----------
    `Paper <https://arxiv.org/abs/1703.05192>`_

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

    def __init__(self, img_shape, generator_cls=GeneratorUNet,
                 discriminator_cls=Discriminator):
        super().__init__()

        self.generator_a = generator_cls(img_shape[0])
        self.generator_b = generator_cls(img_shape[9])

        self.discriminator_a = discriminator_cls(img_shape)
        self.discriminator_b = discriminator_cls(img_shape)

    def forward(self, imgs_a, imgs_b):
        fake_b = self.generator_a(imgs_a)
        rec_a = self.generator_b(fake_b)

        fake_a = self.generator_b(imgs_b)
        rec_b = self.generator_a(fake_a)

        discr_real_a = self.discriminator_a(imgs_a)
        discr_fake_a = self.discriminator_a(fake_a)

        discr_real_b = self.discriminator_b(imgs_b)
        discr_fake_b = self.discriminator_b(fake_b)

        return {"fake_a": fake_a, "fake_b": fake_b, "rec_a": rec_a,
                "rec_b": rec_b, "discr_real_a": discr_real_a,
                "discr_fake_a": discr_fake_a, "discr_real_b": discr_real_b,
                "discr_fake_b": discr_fake_b}

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

        preds = model(data_dict["data"])

        # pixelwise losses
        loss_pixel_a = losses["pixelwise"](preds["fake_a"],
                                           data_dict["data_a"])
        loss_pixel_b = losses["pixelwise"](preds["fake_b"],
                                           data_dict["data_b"])
        loss_pixel = (loss_pixel_a + loss_pixel_b) / 2
        loss_vals["pixelwise"] = loss_pixel.item()
        loss_vals["pixel_A"] = loss_pixel_a.item()
        loss_vals["pixel_B"] = loss_pixel_b.item()

        # adversarial GAN losses
        loss_gan_a = losses["adversarial"](preds["discr_fake_a"], True)
        loss_gan_b = losses["adversarial"](preds["discr_fake_b"], True)
        loss_gan = (loss_gan_a + loss_gan_b) / 2

        loss_vals["gen_adv_A"] = loss_gan_a.item()
        loss_vals["gen_adv_B"] = loss_gan_b.item()
        loss_vals["adversarial"] = loss_gan.item()

        # cycle consistency losses
        loss_cycle_a = losses["cycle"](preds["rec_a"], data_dict["data_a"])
        loss_cycle_b = losses["cycle"](preds["rec_b"], data_dict["data_b"])
        loss_cycle = (loss_cycle_a + loss_cycle_b) / 2

        loss_vals["cycle_A"] = loss_cycle_a.item()
        loss_vals["cycle_B"] = loss_cycle_b.item()
        loss_vals["cycle"] = loss_cycle.item()

        # total generator loss
        loss_gen = loss_cycle + loss_gan + loss_pixel
        loss_vals["generator"] = loss_gen.item()

        # parameter update
        optimizers["generator"].zero_grad()
        loss_gen.backward(retain_graph=False)
        optimizers["generator"].step()

        # Adversarial losses for discriminator A
        loss_real_a = losses["adversarial"](preds["discr_real_a"], True)
        loss_fake_a = losses["adversarial"](preds["discr_fake_a"], False)

        loss_discr_a = (loss_real_a + loss_fake_a) / 2

        loss_vals["discr_real_a"] = loss_real_a.item()
        loss_vals["discr_fake_a"] = loss_fake_a.item()
        loss_vals["discr_a"] = loss_discr_a.item()

        # update discriminator A
        optimizers["discriminator_a"].zero_grad()
        loss_discr_a.backward()
        optimizers["discriminator_a"].step()

        # Adversarial losses for discriminator B
        loss_real_b = losses["adversarial"](preds["discr_real_b"], True)
        loss_fake_b = losses["adversarial"](preds["discr_fake_b"], False)

        loss_discr_b = (loss_real_b + loss_fake_b) / 2

        loss_vals["discr_real_b"] = loss_real_b.item()
        loss_vals["discr_fake_b"] = loss_fake_b.item()
        loss_vals["discr_b"] = loss_discr_b.item()

        # update discriminator B
        optimizers["discriminator_b"].zero_grad()
        loss_discr_b.backward()
        optimizers["discriminator_b"].step()

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
