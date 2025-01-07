import os
import numpy as np
import scipy.io
from numba.typed import Dict
from numba.core import types
import pickle

print(os.getcwd())
print(os.path.abspath('data_var_L2.py'))
import sys
print(sys.path)

from neurolib.model.wc import WCModel

datadir = os.path.join(os.getcwd(), "neurolib", "whole-brain")


from neurolib.models.wc import WCModel
from neurolib.control.optimal_control import oc_wc, cost_functions


def setparams(model):
    model.params.c_excinh = 12  # local E-I coupling
    model.params.signalV = 80.0
    model.params.K_gl = 0.5  # global coupling strength
    model.params.a_exc = 1.0  # excitatory gain
    model.params.a_inh = 1.0  # inhibitory gain
    model.params.mu_exc = 5.0  # excitatory firing threshold
    model.params.mu_inh = 5.0  # inhibitory firing threshold


def get_results(d, wi, mc):
    d["control"][wi] = mc.control.copy()
    d["state"][wi] = mc.get_xs()
    d["energy_input"][wi] = cost_functions.control_strength_cost(mc.control, weights, dt)
    d["sync_cost"][wi] = (mc.compute_total_cost() - d["energy_input"][wi]) / d["weights"][wi]


def dump_data():
    with open(os.path.join(datadir, "wb.pickle"), "wb") as f:
        pickle.dump([data_0, data_1, data_2], f)


def optimize_model(model, step_exp_array, wi, it):
    for k in step_exp_array:
        model.zero_step_encountered = False
        model.step = 10.0 ** (k)
        model.optimize(it)

        get_results(d, wi, model)
        dump_data()


if os.getcwd().split("/")[-1] != "neurolib":
    os.chdir("..")

os.chdir("..")
datadir = os.path.join(os.getcwd(), "neurolib", "whole-brain")

print(datadir)

weights = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)

weights["w_2"] = 1.0
weights["w_1"] = 0.0
weights["w_1T"] = 0.0
weights["w_1D"] = 0.0

N = 100

folders = [
    "NAP_002",
    "NAP_003",
    "NAP_005",
    "NAP_007",
    "NAP_009",
    "NAP_010",
    "NAP_011",
    "NAP_012",
    "NAP_013",
    "NAP_014",
    "NAP_015",
    "NAP_016",
    "NAP_018",
    "NAP_020",
    "NAP_022",
    "NAP_024",
    "NAP_025",
    "NAP_026",
    "NAP_027",
    "NAP_028",
    "NAP_029",
    "NAP_031",
    "NAP_032",
    "NAP_033",
    "NAP_035",
    "NAP_036",
]
cmat_sum = np.zeros((N, N))
dmat_sum = np.zeros((N, N))

for f in folders:
    path = os.path.join(os.getcwd(), "data", "structural_data", "Sch100", f, "SC", "DTI_CM.mat")
    cmat_sum += scipy.io.loadmat(path)["SC"]
    path = os.path.join(os.getcwd(), "data", "structural_data", "Sch100", f, "SC", "DTI_LEN.mat")
    dmat_sum += scipy.io.loadmat(path)["LEN"]

cmat_av = cmat_sum / len(folders)
dmat_av = dmat_sum / len(folders)

with open(os.path.join(datadir, "wb.pickle"), "rb") as f:
    data_read = pickle.load(f)

[data_0, data_1, data_2] = data_read


controlmat = np.zeros((N, 2))
controlmat[:, 0] = 1

costmat = np.zeros((N, 2))
costmat[:, 0] = 1.0

duration = 400.0
target_period = 1

int0 = 0
int1 = None

max_cntrl = 5
pr = np.arange(0, 101, 1)
dt = 0.1


it = 4

for k in range(2):

    for d in [data_0, data_1, data_2]:
        print("###########################################################")

        for wi in range(3):
            print("------------------------------------------------------------ wi = ", wi)

            # if wi not in [2]: continue

            model = WCModel(Cmat=cmat_av, Dmat=dmat_av, seed=0)
            model.params.exc_init = d["init_state"][0]
            model.params.inh_init = d["init_state"][1]
            setparams(model)

            model.params["exc_ext_baseline"] = d["coordinates"][0]
            model.params["inh_ext_baseline"] = d["coordinates"][1]
            model.params.duration = duration
            dt = model.params.dt

            model.run()

            model_controlled = oc_wc.OcWc(
                model,
                target_period,
                print_array=pr,
                cost_interval=(int0, int1),
                control_interval=(int0, int1),
                cost_matrix=costmat,
                control_matrix=controlmat,
            )
            model_controlled.weights["w_p"] = 0.0
            model_controlled.weights["w_2"] = 1.0
            model_controlled.weights["w_var"] = d["weights"][wi]

            model_controlled.maximum_control_strength = max_cntrl

            model_controlled.control = d["control"][wi].copy()
            model_controlled.update_input()
            model_controlled.simulate_forward()
            model_controlled.optimize(0)

            for j in range(1):
                model_controlled.grad_method = 0
                model_controlled.channelwise_optimization = True
                optimize_model(model_controlled, [-2, 0, 2], wi, it)
                model_controlled.channelwise_optimization = False
                optimize_model(model_controlled, [-2, 0, 2], wi, it)

                model_controlled.grad_method = 1
                model_controlled.channelwise_optimization = True
                optimize_model(model_controlled, [-2, 0, 2], wi, it)
                model_controlled.channelwise_optimization = False
                optimize_model(model_controlled, [-2, 0, 2], wi, it)
