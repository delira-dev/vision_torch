from deliravision.models.gans.softmax.models import Discriminator, Generator
from delira.models import AbstractPyTorchNetwork

import torch


class SoftmaxGAN(AbstractPyTorchNetwork):

    """
    Class implementing Softmax GAN

    References
    ----------
    `Paper <https://arxiv.org/abs/1704.06191>`_

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

    def __init__(self, latent_dim, img_shape, generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_shape : tuple
            the shape of the image batches (both generated and real);
            Should include the channels but not the batch dimension
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """

        super().__init__()

        self.generator = generator_cls(latent_dim, img_shape)
        self.discriminator = discriminator_cls(img_shape)

        self._latent_dim = latent_dim

    def forward(self, x, z=None):
        """
        Feeds as single set of batches through all relevant networks

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The image batch
        z : :class:`torch.Tensor`
            the batch of noise vectors, will be sampled if not given

        Returns
        -------
        dict
            dictionary containing all the necessary (intermediate) outputs for
            loss calculation and training

        """
        if z is None:
            z = torch.randn(x.size(0), self._latent_dim, device=x.device,
                            dtype=x.dtype)

        gen_imgs = self.generator(z)

        d_real = self.discriminator(x)
        d_fake = self.discriminator(gen_imgs)

        return {"gen_imgs": gen_imgs, "discr_real": d_real,
                "discr_fake": d_fake}

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

        preds = model(data_dict["data"])

        # partition function
        part_fn = (torch.exp(-preds["discr_real"]).sum()
                   + torch.exp(-preds["discr_fake"]).sum())
        loss_vals["partition_fn"] = part_fn.item()

        # 1 / (batchsize * 2)
        g_target = 1 / (preds["gen_imgs"].size(0) * 2)
        # 1 / batchsize
        d_target = 1 / preds["gen_imgs"].size(0)

        g_loss = (g_target * (preds["discr_real"].sum()
                              + preds["discr_fake"].sum())
                  + (part_fn + 1e-8).log())

        loss_vals["generator"] = g_loss.item()

        optimizers["generator"].zero_grad()
        g_loss.backward(retain_graph=True)
        optimizers["generator"].step()

        d_loss = d_target * preds["discr_real"].sum() + (part_fn + 1e-8).log()
        loss_vals["discriminator"] = d_loss.item()

        optimizers["discriminator"].zero_grad()
        d_loss.backward()
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
            "data": torch.from_numpy(batch["data"]).to(torch.float
                                                       ).to(input_device)}
