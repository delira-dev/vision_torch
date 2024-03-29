from delira.models import AbstractPyTorchNetwork
import torch

from deliravision.models.gans.context_encoder.models import Generator, \
    Discriminator


class ContextEncoder(AbstractPyTorchNetwork):
    """
    Skeleton for Context Encoding Generative Adversarial Networks with
    exchangeable Generator and Discriminator classes

    This GAN is suitable for unsupervised visual feature learning driven by
    context-based pixel prediction and empowers an unsupervised training
    approach based on in-painting.

    References
    ----------
    `Paper <https://arxiv.org/abs/1604.07379>`_

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
    def __init__(self, in_channels,
                 generator_cls=Generator,
                 discriminator_cls=Discriminator):
        """

        Parameters
        ----------
        in_channels : int
            number of image channels
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology

        """
        super().__init__()

        self.generator = generator_cls(in_channels)
        self.discriminator = discriminator_cls(in_channels)

    def forward(self, x, masked_imgs=None, masked_parts=None):
        """
        Feeds an image batch (and masks if given) through all necessary networks

        Parameters
        ----------
        x : :class:`torch.Tensor`
             the image batch
        masked_imgs : :class:`torch.Tensor`
            the masked images; will be generated by applying a random mask if
            None
        masked_parts : :class:`torch.Tensor`
            the parts of the images which were masked; Will be extracted during
            application of the random mask, if not provided

        Returns
        -------
        dict
            a dictionary containing all (intermediate) outputs necessary for
            loss calculation and training

        """

        # mask images if necessary
        if masked_imgs is None or masked_parts is None:
            masked_imgs, masked_parts = self.apply_random_mask(x)

        gen_parts = self.generator(masked_imgs)

        discr_fake = self.discriminator(gen_parts)
        discr_real = self.discriminator(masked_parts)

        return {"gen_parts": gen_parts, "discr_fake": discr_fake,
                "discr_real": discr_real, "masked_imgs": masked_imgs,
                "masked_parts": masked_parts}

    @staticmethod
    def apply_mask(imgs: torch.Tensor, mask: torch.Tensor, mask_size: int):
        """

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch
        mask : :class:`torch.Tensor`
            the mask to apply
        mask_size : int
            the size of the mask (needed for index computation)

        Returns
        -------
        :class:`torch.Tensor`
            the masked image batch

        """
        masked_imgs = imgs.clone()
        img_parts = torch.empty(imgs.size(0), imgs.size(1), mask_size,
                                mask_size, device=imgs.device,
                                dtype=imgs.dtype)

        for i, (y1, x1) in enumerate(mask):
            y2, x2 = y1 + mask_size, x1 + mask_size
            masked_imgs[i, :, y1:y2, x1:x2] = 1
            img_parts[i, ...] = imgs[i, :, y1:y2, x1:x2]

        return masked_imgs, img_parts

    @staticmethod
    def _generate_random_mask(batchsize, max_val):
        """
        Generates a random mask

        Parameters
        ----------
        batchsize : int
            the number of masks to generate
        max_val : int
            the maximum index value for a mask

        Returns
        -------
        :class:`torch.Tensor`
            the indices to mask

        """
        return torch.randint(max_val, (batchsize, 2))

    def apply_random_mask(self, imgs):
        """
        Generates a random mask and directly applies it on a given image batch

        Parameters
        ----------
        imgs : :class:`torch.Tensor`
            the image batch

        Returns
        -------
        :class:`torch.Tensor`
            the masked image batch

        """
        mask = self._generate_random_mask(imgs.size(0),
                                          self._img_size - self._mask_size)

        return self.apply_mask(imgs, mask, self._mask_size)

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

        pixel_loss = losses["pixelwise"](preds["gen_parts"],
                                         preds["masked_parts"])
        adv_loss = losses["adversarial"](preds["discr_fake"], True)

        gen_loss = 0.001 * adv_loss + 0.999 * pixel_loss

        loss_vals["gen_pixelwise"] = pixel_loss.item()
        loss_vals["gen_adversarial"] = adv_loss.item()
        loss_vals["generator"] = gen_loss.item()

        optimizers["generator"].zero_grad()
        gen_loss.backward(retain_graph=True)
        optimizers["generator"].step()

        discr_loss_fake = losses["adversarial"](preds["discr_fake"], False)
        discr_loss_real = losses["adversarial"](preds["discr_real"], True)

        discr_loss = (discr_loss_fake + discr_loss_real) / 2

        loss_vals["discr_fake"] = discr_loss_fake.item()
        loss_vals["discr_real"] = discr_loss_real.item()
        loss_vals["discriminator"] = discr_loss.item()

        optimizers["discriminator"].zero_grad()
        discr_loss.step()
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
