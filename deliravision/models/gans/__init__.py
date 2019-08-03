from deliravision.models.gans.adversarial_autoencoder import *
from deliravision.models.gans.auxiliary_classifier import *
from deliravision.models.gans.boundary_equilibrium import *
from deliravision.models.gans.boundary_seeking import *
from deliravision.models.gans.conditional import *
from deliravision.models.gans.context_conditional import *
from deliravision.models.gans.context_encoder import *
from deliravision.models.gans.coupled import *
from deliravision.models.gans.cycle import *
from deliravision.models.gans.deep_convolutional import *
from deliravision.models.gans.disco import *
from deliravision.models.gans.dragan import *
from deliravision.models.gans.dual import *
from deliravision.models.gans.energy_based import *
from deliravision.models.gans.enhanced_super_resolution import *
from deliravision.models.gans.gan import *
from deliravision.models.gans.info import *
from deliravision.models.gans.munit import *
from deliravision.models.gans.pix2pix import *
from deliravision.models.gans.pixel_da import *
from deliravision.models.gans.relativistic import *
from deliravision.models.gans.semi_supervised import *
from deliravision.models.gans.softmax import *
from deliravision.models.gans.star import *
from deliravision.models.gans.super_resolution import *
from deliravision.models.gans.unit import *
from deliravision.models.gans.wasserstein import *
from deliravision.models.gans.wasserstein_gp import *
from deliravision.models.gans.wasserstein_div import *

# make LSGAN a synonym for basic GAN, since training only differs in loss
# function, which isn't specified here
LeastSquareGAN = GenerativeAdversarialNetworks
