import torch


class AdversarialLoss(torch.nn.Module):
    def __init__(self, loss_fn=torch.nn.BCELoss(), same_size=False):
        #TODO: docstring
        super().__init__()
        self._loss_fn = loss_fn
        self._same_size = same_size

    def forward(self, pred: torch.Tensor, target: bool):
        if self._same_size:
            gt = torch.ones_like(pred)
        else:
            gt = torch.ones(pred.size(0), 1, device=pred.device,
                            dtype=pred.dtype)

        gt = gt * int(target)

        return self._loss_fn(pred, gt)


class IterativeDiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, gt):
        return sum([torch.mean((out - gt) ** 2) for out in preds])


class GradientPenalty(torch.nn.Module):
    """
    A module to compute the gradient penalty
    """

    def forward(self, discr_interpolates: torch.Tensor,
                interpolates: torch.Tensor):
        """
        Computes the gradient penalty
        Parameters
        ----------
        discr_interpolates : :class:`torch.Tensor`
            the discriminator's output for the :param:`interpolates`
        interpolates : :class:`torch.Tensor`
            randomly distorted images as input for the discriminator
        Returns
        -------
        :class:`torch.Tensor`
            a weighted gradient norm
        """

        fake = torch.ones(interpolates.size(0), 1, device=interpolates.device,
                          dtype=interpolates.dtype)

        gradients = torch.autograd.grad(
            outputs=discr_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        return ((gradients.norm(p=2, dim=1) - 1) ** 2).mean()


class PullAwayLoss(torch.nn.Module):
    """
    Pull Away Loss for the Energy-Based GANs
    References
    ----------
    `Paper <https://arxiv.org/abs/1609.03126>`_
    """

    def __init__(self, weight=1.):
        """
        Parameters
        ----------
        weight : float
            weight factor (specifying the impact compared to other loss
            functions)
        """
        super().__init__()
        self._weight = weight

    def forward(self, embeddings: torch.Tensor):
        """
        Parameters
        ----------
        embeddings : :class:`torch.Tensor`
            the embeddings of image batches
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        norm = (embeddings ** 2).sum(-1, keepdim=True).sqrt()
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb,
                                  normalized_emb.transpose(0, 1))
        batchsize = embeddings.size(0)

        pt_loss = ((similarity.sum() - batchsize)
                   / (batchsize * (batchsize - 1)))

        return pt_loss * self._weight


class DiscriminatorMarginLoss(torch.nn.Module):
    """
    A loss whose calculation switches slightly depending on a calculated
    margin.
    References
    --------
    `Paper <https://arxiv.org/abs/1609.03126>`_
    """

    def __init__(self, divisor=64., loss_fn=torch.nn.MSELoss()):
        super().__init__()
        self._divisor = divisor
        self._loss_fn = loss_fn

    def forward(self, real_recon, real_imgs, fake_recon, fake_imgs):
        """
        Calculates the loss out of the given parameters
        Parameters
        ----------
        real_recon : :class:`torch.Tensor`
            the reconstruction of the real images
        real_imgs : :class:`torch.Tensor`
            the real image batch
        fake_recon : :class:`torch.Tensor`
            the reconstruction of the fake images
        fake_imgs : :class:`torch.Tensor`
            the (generated) fake image batch
        Returns
        -------
        :class:`torch.Tensor`
            the combined (margin-dependent) loss for real and fake images
        :class:`torch.Tensor`
            the loss only for real images
        :class:`torch.Tensor`
            the loss only for fake images
        """
        discr_real = self._loss_fn(real_recon, real_imgs)
        discr_fake = self._loss_fn(fake_recon, fake_imgs)

        margin = max(1., real_imgs.size(0) / self._divisor)

        discr_loss = discr_real

        if (margin - discr_fake).item() > 0:
            discr_loss += margin - discr_fake

        return discr_loss, discr_real, discr_fake


class BELoss(torch.nn.Module):
    """
    Boundary Equilibrium Loss
    """

    def __init__(self, gamma=0.75, lambda_k=0.001, initial_k=0.0):
        """
        Parameters
        ----------
        gamma : float
            impact of real_loss on weight update
        lambda_k : float
            impact of loss difference on weight update
        initial_k : float
            initial weight value
        """
        super().__init__()
        self._k = initial_k
        self._gamma = gamma
        self._lambda_k = lambda_k

    def forward(self, discr_real, real_imgs, discr_fake, gen_imgs):
        """
        Computes the losses
        Parameters
        ----------
        discr_real : :class:`torch.Tensor`
            the discriminiators output for real images
        real_imgs : :class:`torch.Tensor`
            the real images
        discr_fake : :class:`torch.Tensor`
            the discriminator output for generated images
        gen_imgs : :class:`torch.Tensor`
            the generated images
        Returns
        -------
        :class:`torch.Tensor`
            the total loss
        :class:`torch.Tensor`
            the part of the total loss coming from real images
            (without weighting)
        :class:`torch.Tensor`
            the part of the loss coming from fake images (without weighting)
        """
        loss_real = self._loss_fn(discr_real, real_imgs)
        loss_fake = self._loss_fn(discr_fake, gen_imgs.detach())

        total_loss = loss_real - self._k * loss_fake

        # update weight term
        diff = (self._gamma * loss_real - loss_fake).mean().item()
        self._k = self._k + self._lambda_k * diff
        # constrain to [0, 1]
        self._k = min(max(self._k, 0), 1)

        return total_loss, loss_real, loss_fake

    @staticmethod
    def _loss_fn(pred, label):
        """
        The actual loss function; Computes Mean L1-Error
        Parameters
        ----------
        pred : :class:`torch.Tensor`
            the predictions
        label : :class:`torch.Tensor`
            the labels
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        return (pred - label).abs().mean()


class BoundarySeekingLoss(torch.nn.Module):
    """
    Boundary Seeking Loss
    References
    ----------
    https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
    """

    def __init__(self, weight=0.5):
        """
        Parameters
        ----------
        weight : float
            weighting factor
        """
        super().__init__()

        self._weight = weight

    def forward(self, pred):
        """
        Calculates the actual loss
        Parameters
        ----------
        pred : :class:`torch.Tensor`
            the prediction (typically obtained by the discriminator)
        Returns
        -------
        :class:`torch.Tensor`
            the loss value
        """
        return self._weight * torch.mean((torch.log(pred) -
                                          torch.log(1 - pred)) ** 2)


class WassersteinDivergence(torch.nn.Module):
    """
    Implements the Wasserstein Divergence proposed in
    `Wasserstein Divergence for GANS <https://arxiv.org/abs/1712.01026>`_
    """

    def __init__(self, p=6, k=2):
        """
        Parameters
        ----------
        p : int
            order of the norm
        k : int
            multiplicative factor applied to the mean
        """
        super().__init__()
        self._p = p
        self._k = k

    def forward(self, real_imgs, real_val, fake_imgs, fake_val):
        """
        Computes the actual divergence
        Parameters
        ----------
        real_imgs : :class:`torch.Tensor`
            the batch of real images
        real_val : :class:`torch.Tensor`
            the validity results for the real images obtained by feeding them
            through a discriminator
        fake_imgs : :class:`torch.Tensor`
            the batch of generated fake images
        fake_val : :class:`torch.Tensor`
            the validity results of the fake images obtained by feeding them
            through a discriminator
        Returns
        -------
        :class:`torch.Tensor`
            the wasserstein divergence
        """

        real_grad = torch.autograd.grad(
            real_val, real_imgs, torch.ones(real_imgs.size(0), 1,
                                            device=real_imgs.device,
                                            dtype=real_imgs.dtype,
                                            requires_grad=True),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        real_grad_norm = real_grad.norm(p=self._p)

        fake_grad = torch.autograd.grad(
            fake_val, fake_imgs, torch.ones(fake_imgs.size(0), 1,
                                            device=fake_imgs.device,
                                            dtype=fake_imgs.dtype,
                                            requires_grad=True),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]

        fake_grad_norm = fake_grad.norm(p=self._p)

        return torch.mean(real_grad_norm + fake_grad_norm) * self._k / 2
