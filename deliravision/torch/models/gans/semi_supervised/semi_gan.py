from deliravision.models.gans.semi_supervised.models import Discriminator, \
    Generator

from delira.models import AbstractPyTorchNetwork
import torch


class SemiSupervisedGAN(AbstractPyTorchNetwork):
    """
    Class implementing Semi-Supervised Learning with Generative Adversarial
    Networks

    References
    ----------
    `Paper <https://arxiv.org/abs/1606.01583>`_

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

    def __init__(self, latent_dim, img_size, num_channels, num_classes=10,
                 generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        latent_dim : int
            size of the latent dimension
        img_size : int
            number of pixels per side of the image
        num_channels : int
            number of image channels
        num_classes : int
            number of image classes
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """

        super().__init__()

        self.generator = generator_cls(latent_dim, img_size, num_channels,
                                       num_classes)

        self.discriminator = discriminator_cls(img_size, num_channels,
                                               num_classes)

        self._latent_dim = latent_dim
        self._n_classes = num_classes

    def forward(self, x, z=None):
        """
        Feeds a single set of batches through the network

        Parameters
        ----------
        x : :class:`torch.Tensor`
            the image batch
        z : :class:`torch.Tensor`

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

        val_fake, labels_fake = self.discriminator(gen_imgs)
        val_real, labels_real = self.discriminator(x)

        fake_gt_label = torch.randint(self._n_classes, x.size(0),
                                      device=gen_imgs.device)

        return {"gen_imgs": gen_imgs, "val_fake": val_fake,
                "labels_fake": labels_fake, "val_real": val_real,
                "labels_real": labels_real, "fake_gt_label": fake_gt_label}

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

        gen_loss = losses["adversarial"](preds["val_fake"], True)
        loss_vals["generator"] = gen_loss.item()

        optimizers["generator"].zero_grad()
        gen_loss.backward(retain_graph=True)
        optimizers["generator"].step()

        discr_fake_adv = losses["adversarial"](preds["val_fake"], False)
        discr_fake_aux = losses["auxiliary"](preds["labels_fake"],
                                             preds["fake_gt_label"])

        discr_fake = (discr_fake_adv + discr_fake_aux) / 2

        discr_real_adv = losses["adversarial"](preds["val_real"], True)
        discr_real_aux = losses["auxiliary"](preds["labels_real"],
                                             data_dict["label"])

        discr_real = (discr_real_adv + discr_real_aux) / 2

        discr_loss = (discr_fake + discr_real) / 2

        loss_vals.update({
            "discr_fake_adv": discr_fake_adv.item(),
            "discr_fake_clf": discr_fake_aux.item(),
            "discr_real_adv": discr_real_adv.item(),
            "discr_real_clf": discr_real_aux.item(),
            "discr_fake": discr_fake.item(),
            "discr_real": discr_real.item(),
            "discriminator": discr_loss.item()
        })

        with torch.no_grad():
            pred_all = torch.cat((preds["labels_real"], preds["labels_fake"]),
                                 dim=0)
            gt_all = torch.cat((data_dict["label"], preds["fake_gt_label"]))

            acc = torch.mean((torch.argmax(pred_all, dim=1) == gt_all).float())

            metric_vals["accuary"] = acc.item()

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
        return {
            "data": torch.from_numpy(batch["data"]).to(torch.float
                                                       ).to(input_device),
            "label": torch.from_numpy(batch["label"]).squeeze(-1).to(
                torch.float).to(output_device)
        }
