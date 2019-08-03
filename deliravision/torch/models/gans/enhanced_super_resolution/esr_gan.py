from delira.models import AbstractPyTorchNetwork
from deliravision.models.gans.enhanced_super_resolution.models import Discriminator, GeneratorRRDB, \
    FeatureExtractor
import torch


class EnhancedSuperResolutionGAN(AbstractPyTorchNetwork):
    """
    Class implementing the
    Enhanced Super Resolution Generative Adversarial Networks.

    References
    ----------
    `Paper <https://arxiv.org/abs/1809.00219>`_

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

    def __init__(self, input_shape, num_filters=64, num_res_blocks=6,
                 num_upsample=2, generator_cls=GeneratorRRDB,
                 discriminator_cls=Discriminator,
                 feature_extractor_cls=FeatureExtractor):
        """

        Parameters
        ----------
        input_shape : tuple
            shape of the image batches (including channels,
            excluding batch dimension)
        num_filters : int
            number of initial filters
        num_res_blocks : int
            number of residual blocks in the generator
        num_upsample : int
            number of upsampling steps in the generator
        generator_cls :
            class implementing the actual generator topology
        discriminator_cls :
            class implementing the actual discriminator topology
        feature_extractor_cls :
            class wrapping a pre-trained network trained on imagenet for
            feature comparisoon; defaults to a VGG19
        """
        super().__init__()

        self.generator = generator_cls(input_shape[0], num_filters,
                                       num_res_blocks, num_upsample)

        self.discriminator = discriminator_cls(input_shape)
        self.feature_extractor = feature_extractor_cls()

    def train(self, mode=True):
        """
        Overloads the ``train`` function of :class:`torch.nn.Module` to
        keep the feature extractor in eval mode always

        Parameters
        ----------
        mode : bool
            the train mode

        Returns
        -------
        :class:`EnhancedSuperResolutionGAN`
            the self reference

        """
        self.generator.train(mode)
        self.discriminator.train(mode)
        # keep feature extractor in eval mode
        self.feature_extractor.eval()
        return self

    def forward(self, imgs_lr: torch.Tensor, imgs_hr: torch.Tensor):
        """
        Forwards a batch of High- and Low- Resolution images through the
        networks

        Parameters
        ----------
        imgs_lr : :class:`torch.Tensor`
            the low-resolution image batch
        imgs_hr : :class:`torch.Tensor`
            the high resolution image batch

        Returns
        -------
        dict
            a dictionary containing all the (intermediate) outputs necessary
            for loss calculation and training

        """
        gen_imgs = self.generator(imgs_lr)
        discr_real = self.discriminator(imgs_hr)
        discr_fake = self.discriminator(gen_imgs)

        features_real = self.feature_extractor(imgs_hr)
        features_gen = self.feature_extractor(gen_imgs)

        return {"gen_imgs": gen_imgs, "discr_real": discr_real,
                "discr_fake": discr_fake, "features_real": features_real,
                "features_fake": features_gen}

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

        preds = model(data_dict["data_lr"], data_dict["data_hr"])

        # pixelwise loss
        loss_pixel = losses["pixelwise"](preds["gen_imgs"],
                                         data_dict["data_hr"])
        loss_vals["gen_pixelwise"] = loss_pixel.item()

        # Adversarial loss (relativistic average GAN)
        loss_adv = losses["adversarial"](
            preds["discr_fake"]-preds["discr_real"].mean(0, keepdim=True),
            True)
        loss_vals["gen_adversarial"] = loss_adv.item()

        # Content Loss
        loss_content = losses["content"](preds["features_fake"],
                                         preds["features_real"])
        loss_vals["gen_content"] = loss_content.item()

        loss_gen = (loss_content + attr_module.lambda_adversarial * loss_adv
                    + attr_module.lambda_pixel * loss_pixel)
        loss_vals["generator"] = loss_gen.item()

        # update generator
        optimizers["generator"].zero_grad()
        loss_gen.backward(retain_graph=True)
        optimizers["generator"].step()

        # discriminator losses (relativistic average GAN)
        loss_real = losses["adversarial"](
            preds["discr_real"] - preds["discr_fake"].mean(0, keepdim=True),
            True)
        loss_vals["discr_real"] = loss_real.item()
        loss_fake = losses["adversarial"](
            preds["discr_fake"] - preds["discr_real"].mean(0, keepdim=True),
            False)
        loss_vals["discr_fake"] = loss_fake.item()

        loss_discr = (loss_real + loss_fake) / 2
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
            "data_lr": torch.from_numpy(batch["data_lr"]).to(
                torch.float).to(input_device),
            "data_hr": torch.from_numpy((batch["data_hr"]).to(
                torch.float).to(input_device))
        }
