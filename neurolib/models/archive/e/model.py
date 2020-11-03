import numpy as np

from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class Model_e(Model):
    """
    Multi-population mean-field model with exciatory and inhibitory neurons per population.
    """

    name = "e"
    description = "Simplified version of simplified aln for testing purposes"

    init_vars = [
        "rates_exc_init",
        "mufe_init",
        "seem_init",
        "seev_init",
    ]

    state_vars = [
        "rates_exc",
        "mufe",
        "seem",
        "seev",
        "sigmae_f",
        "tau_exc",
    ]
    output_vars = ["rates_exc", "mufe", "seem", "seev", "sigmae_f", "tau_exc"]
    default_output = "rates_exc"
    target_output_vars = ["rates_exc"]
    input_vars = ["ext_exc_current"]
    default_input = "ext_exc_current"
    control_input_vars = ["ext_exc_current"]
    

    def __init__(self, params=None, Cmat=None, Dmat=None, lookupTableFileName=None):
        """
        :param params: parameter dictionary of the model
        :param Cmat: Global connectivity matrix (connects E to E)
        :param Dmat: Distance matrix between all nodes (in mm)
        :param lookupTableFileName: Filename for precomputed transfer functions and tables
        :param seed: Random number generator seed
        :param simulateChunkwise: Chunkwise time integration (for lower memory use)
        """

        self.Cmat = Cmat  # Connectivity matrix
        self.Dmat = Dmat  # Delay matrix
        self.lookupTableFileName = lookupTableFileName  # Filename for aLN lookup functions

        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(
                Cmat=self.Cmat, Dmat=self.Dmat, lookupTableFileName=self.lookupTableFileName
            )

        # Initialize base class Model
        super().__init__(integration=integration, params=params)