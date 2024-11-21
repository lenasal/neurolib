from . import loadDefaultParams as dp
from . import timeIntegration as ti

from neurolib.models.model import Model


class LinOscModel(Model):

    name = "linosc"
    description = "Kuramoto Model"

    init_vars = ["theta_init"]
    state_vars = ["theta"]
    output_vars = ["theta"]
    default_output = "theta"
    input_vars = ["theta_ext"]
    default_input = "theta_ext"

    def __init__(self, params=None, Cmat=None, Dmat=None, seed=None):
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.seed = seed

        integration = ti.timeIntegration

        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat, seed=self.seed)

        super().__init__(params=params, integration=integration)
