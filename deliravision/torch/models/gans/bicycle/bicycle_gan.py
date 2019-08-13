from deliravision.models.gans.bicycle.models import \
    Generator, Encoder, MultiResolutionDiscriminator
from delira.models import AbstractPyTorchNetwork
import torch


class BiCycleGAN(AbstractPyTorchNetwork):
    """
    A PyTorch Impelementation of the bycicle GAN

    It consists of a generator learning to map the given input (combined with
    its latent code) to the output while explicitly encouraging the connection
    between output and latent code to be invertible to prevent mode collapse
    and to produce more diverse results.

    References
    ----------
    `Paper <https://arxiv.org/abs/1711.11586>`_

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

    def __init__(self, latent_dim, img_shape,
                 generator_cls=Generator, encoder_cls=Encoder,
                 discriminator_cls=MultiResolutionDiscriminator,
                 lambda_pixel=10., lambda_latent=0.5, lambda_kl=0.01):
        """
        Parameters
        ----------
        latent_dim : int
            the size of the latent dimension
        img_shape : tuple
            the shape of the input images (including channel-dimension,
            excluding batch-dimension)
        generator_cls :
            class/function implementing the actual generator topology
        encoder_cls :
            class/function implementing the actual encoder topology
        discriminator_cls :
            class/function implementing the actual discriminator topology
        lambda_pixel : float
            weight for the pixlewise loss
        lambda_latent : float
            weight for the latent loss
        lambda_kl : float
            weight for the kl divergence
        """
        super().__init__()

        self._latent_dim = latent_dim

        self.generator = generator_cls(latent_dim, img_shape)
        self.encoder = encoder_cls(latent_dim)

        self.discr_vae = discriminator_cls(img_shape[0])
        self.discr_lr = discriminator_cls(img_shape[0])

        self.lambda_pixel = lambda_pixel
        self.lambda_latent = lambda_latent
        self.lambda_kl = lambda_kl

    @staticmethod
    def reparametrize(mu, logvar):
        """
        Reparametrizes the data from a normal distribution
        Parameters
        ----------
        mu : :class:`torch.Tensor`
            the mean vector to use for a multivariate normal distribution
        logvar : :class:`torch.Tensor`
            the logvars, to calculate the stds for the multivariate normal
            distribution from
        Returns
        -------
        :class:`torch.Tensor`
            the reparametrized vector
        """
        std = torch.exp(logvar / 2)
        return torch.normal(mu, std)

    def forward(self, real_a: torch.Tensor, real_b: torch.Tensor):
        """
        Forwards the given tensors (and the intermediate outputs of other
        networks) through all necessary networks and returns all values to
        calculate losses
        Parameters
        ----------
        real_a : class:`torch.Tensor`
            the input of the first (source) domain
        real_b : :class:`torch.Tensor`
            the input of the second (target) domain
        Returns
        -------
        dict of :class:`torch.Tensor`
            a dictionary containing all necessary results for loss calculation
            and training
        """

        real_mu, real_log_var = self.encoder(real_b)
        encoded_z = self.reparametrize(real_mu, real_log_var)

        fake_b = self.generator(real_a, encoded_z)

        sampled_z = torch.rand(real_a.size(0),
                               self._latent_dim,
                               device=real_a.device,
                               dtype=real_a.dtype)

        sampled_b = self.generator(real_a, sampled_z)

        fake_mu, fake_log_var = self.encoder(fake_b)

        vae_real = self.discr_vae(real_b)
        vae_fake = self.discr_vae(fake_b)

        lr_real = self.discr_lr(real_b)
        lr_fake = self.discr_lr(fake_b)

        return {"real_mu": real_mu,
                "real_log_var": real_log_var,
                "encoded_z": encoded_z,
                "fake_mu": fake_mu,
                "fake_log_var": fake_log_var,
                "sampled_z": sampled_z,
                "sampled_b": sampled_b,
                "fake_b": fake_b,
                "vae_real": vae_real,
                "vae_fake": vae_fake,
                "lr_fake": lr_fake,
                "lr_real": lr_real}

    @staticmethod
    def closure(model, data_dict: dict, optimizers: dict, losses=None,
                metrics=None, fold=0, **kwargs):
        """
        Function which handles prediction from batch, logging, loss calculation
        and optimizer step
        Parameters
        ----------
        model : :class:`AbstractPyTorchNetwork`
            model to forward data through
        data_dict : dict
            dictionary containing the data
        optimizers : dict
            dictionary containing all optimizers to perform parameter update
        losses : dict
            Functions or classes to calculate losses
        metrics : dict
            Functions or classes to calculate other metrics; won't be used here
        fold : int
            Current Fold in Crossvalidation (default: 0)
        kwargs : dict
            additional keyword arguments
        Returns
        -------
        dict
            Metric values (with same keys as input dict metrics)
        dict
            Loss values (with same keys as input dict losses)
        dict
            Arbitrary number of predictions
        """

        if isinstance(model, torch.nn.DataParallel):
            attr_module = model.module
        else:
            attr_module = model

        loss_vals, metric_vals = {}, {}

        # use data as real A and label as real B
        preds = model(data_dict["data"], data_dict["label"])

        # Pixelwise loss of translated img
        loss_pixel = losses["pixelwise"](preds["fake_b"], data_dict["label"])
        loss_vals["pixelwise"] = loss_pixel.item()

        # KL-Divergence of encoded B
        loss_kl = losses["kl"](preds["real_mu"], preds["real_log_var"])
        loss_vals["kl"] = loss_kl.item()

        # Adversarial Loss
        loss_vae = losses["gan_mse"](preds["fake_b"], True)
        loss_vals["vae_gen"] = loss_vae.item()

        # loss cLR-GAN
        loss_lr = losses["gan_mse"](preds["sampled_b"], True)
        loss_vals["lr_gen"] = loss_lr.item()

        # calculate common loss for generator and encoder
        loss_ge = (loss_vae + loss_lr +
                   getattr(attr_module, "lambda_pixel") * loss_pixel +
                   getattr(attr_module, "lambda_kl") * loss_kl)
        loss_vals["total_encoder"] = loss_ge.item()

        # zero gradients for encoder and generator
        # (no accumulation over iterations)
        optimizers["generator"].zero_grad()
        optimizers["encoder"].zero_grad()

        # backward common loss for encoder and decoder and retain graph for
        # later generator update
        loss_ge.backward(retain_graph=True)
        # update encoder
        optimizers["encoder"].step()

        # calculate l1 loss on latent space
        loss_latent = losses["l1"](
            preds["fake_mu"], preds["sampled_z"]
        )
        loss_vals["latent_gen"] = loss_latent.item()

        # weight loss before backprop to weight gradients w.r.t other already
        # backpropped losses
        loss_latent = loss_latent * getattr(attr_module, "lambda_latent")

        # backward latetnt space loss as additional loss for generator only
        loss_latent.backward()
        # update encoder
        optimizers["generator"].step()

        loss_vals["total_generator"] = loss_latent.item() + loss_ge.item()

        # calculate losses for vae discriminator
        loss_discr_vae_real = losses["gan_mse"](preds["vae_real"], True)
        loss_discr_vae_fake = losses["gan_mse"](preds["vae_fake"], False)

        loss_discr_vae = loss_discr_vae_fake + loss_discr_vae_real
        loss_vals["vae_discr_real"] = loss_discr_vae_real.item()
        loss_vals["vae_discr_fake"] = loss_discr_vae_fake.item()
        loss_vals["vae_discr_total"] = loss_discr_vae.item()

        # update vae discriminator
        optimizers["discriminator_vae"].zero_grad()
        loss_discr_vae.backward()
        optimizers["discriminator_vae"].step()

        # calculate losses for lr discriminator
        loss_discr_lr_real = losses["gan_mse"](preds["lr_real"], True)
        loss_discr_lr_fake = losses["gan_mse"](preds["lr_fake"], False)

        loss_discr_lr = loss_discr_lr_fake + loss_discr_lr_real
        loss_vals["lr_discr_real"] = loss_discr_lr_real.item()
        loss_vals["lr_discr_fake"] = loss_discr_lr_fake.item()
        loss_vals["lr_discr_total"] = loss_discr_lr.item()

        # update lr discriminator
        optimizers["discriminator_lr"].zero_grad()
        loss_discr_lr.backward()
        optimizers["discriminator_lr"].step()

        # zero gradients again just to make sure, gradients aren't carried to
        # next iteration (won't affect training since gradients are zeroed
        # before every backprop step, but would result in way higher memory
        # consumption)
        for k, v in optimizers.items():
            v.zero_grad()

        # return values
        return metric_vals, loss_vals, {k: v.detach()
                                        for k, v in preds.items()}

    @staticmethod
    def prepare_batch(batch: dict, input_device, output_device):
        return {"data": batch["data"].to(torch.float).to(input_device),
                "label": batch["label"].to(torch.float).to(output_device)}
