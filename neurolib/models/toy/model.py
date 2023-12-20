from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class ToyModel(Model):
    """
    Toy model
    """

    name = "toy"
    description = "Toy model"

    init_vars = ["x_init", "y_init"]
    state_vars = ["x", "y"]
    output_vars = ["x", "y"]
    default_output = "x"
    input_vars = ["x_ext", "y_ext"]
    default_input = "x_ext"

    def __init__(self, params=None, Cmat=None, Dmat=None):
        self.Cmat = Cmat
        self.Dmat = Dmat

        # the integration function must be passed
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(Cmat=self.Cmat, Dmat=self.Dmat)

        # Initialize base class Model
        super().__init__(integration=integration, params=params)
