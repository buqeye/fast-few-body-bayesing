from ruamel.yaml import YAML
import schwimmbad
from argparse import ArgumentParser

import os
from os.path import join, basename, splitext
import sys
import shutil
import pickle

import numpy as np
import time
from fit3bf.sampling import ObservableDistribution, TruncationDistribution
from fit3bf.utils import dict_hash
from fit3bf.observables import from_hdf5
import emcee
from sklearn.utils import check_random_state
import datetime
import h5py
import logging


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))


####################################################################
# Parse arguments, e.g.,
# python sample_emulators.py --parameters=parameters/sampling.yaml
# or
# mpiexec -n 8 python sample_emulators.py --parameters=parameters/sampling.yaml --mpi
####################################################################
parser = ArgumentParser(
    description="An MCMC sampler that is sped up via EVC emulators."
)

parser.add_argument(
    "--parameters",
    dest="parameters_file",
    type=str,
    help="The parameters.yaml file that holds all input info",
)
parser.add_argument(
    "--observables",
    dest="observables",
    default=None,
    nargs='+',
    required=False,
    # type=list,
    help="The observables to fit",
)
parser.add_argument(
    "--label",
    dest="label",
    type=str,
    default=None,
    help="The label for the samples directory. Will default to current timestamp.",
)
parser.add_argument(
    "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
)
args = parser.parse_args()
parameters_file = args.parameters_file

yaml = YAML(typ="safe")
with open(parameters_file, "r") as input_file:
    input_parameters = yaml.load(input_file)

fit_obs = []
fit_obs_str = ""

for obs in input_parameters["observables"]:
    if args.observables is not None:
        if obs in args.observables:
            input_parameters["observables"][obs]["fit"] = True
        else:
            input_parameters["observables"][obs]["fit"] = False
    if input_parameters["observables"][obs]["fit"]:
        fit_obs.append(obs)
        fit_obs_str += obs + "-"
fit_obs_str = fit_obs_str[:-1]
print(fit_obs, fit_obs_str)

input_hash = dict_hash(input_parameters)
param_basename = splitext(basename(parameters_file))[0]
if args.label is None:
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
else:
    now = args.label

####################################################################
# Set up ObservableDistribution object, which holds
# all observable classes to make predictions, and the data
# necessary to fit the parameters
####################################################################
data = []
data_stdv = []
observables = []

object_directory = input_parameters["mcmc"]["object_directory"]
observable_names = []
for obs_name, obs in input_parameters["observables"].items():
    if obs["fit"]:
        data.append(obs["experiment_central_value"])
        data_stdv.append(obs["experiment_standard_deviation"])
        observable_names.append(obs_name)
        observables.append(from_hdf5(join(object_directory, f"{obs_name}.h5")))
        # with open(join(object_directory, obs_name + ".pickle"), "rb") as file:
        #     observables.append(pickle.load(file))
cov_expt = np.diag(data_stdv) ** 2

hyp = input_parameters["hyperparameters"]
mcmc = input_parameters["mcmc"]
output_directory = join(object_directory, fit_obs_str + "_" + now)
output_file = join(output_directory, mcmc["output_file"])
parameter_names = mcmc["parameter_names"]
post_parameter_names = mcmc["posterior_parameter_names"]
use_post_parameters = (post_parameter_names is not None) and (len(post_parameter_names) != 0)

if use_post_parameters:
    posterior_lecs_indices = np.where(np.isin(parameter_names, post_parameter_names))[0]
else:
    posterior_lecs_indices = None
print(posterior_lecs_indices)

if not use_post_parameters:
    loc_post_lecs = None
    cov_post_lecs = None
else:
    with open(hyp["mean_file"], "r") as input_file:
        post_param_map = yaml.load(input_file)
        loc_post_lecs = np.array([post_param_map[name] for name in post_parameter_names])
    cov_post_lecs = np.load(hyp["posterior_parameters_covariance_file"], allow_pickle=False)

# cov_post_lecs = None
# if hyp["posterior_parameters_covariance_file"] is not None and hyp["use_gradients"]:
#     print("Propagating posterior parameter uncertainties")
#     cov_post_lecs = np.load(
#         hyp["fixed_parameters_covariance_file"], allow_pickle=False
#     )

# np.seterr(all='raise')
# import warnings
# warnings.filterwarnings('error')

y_ref = hyp["y_ref"]
truncation_ignore_observables = hyp["truncation_ignore_observables"]
trunc_dist_obs_mask = np.ones(len(observable_names), dtype=bool)
if y_ref == "expt":
    y_ref = np.array(np.abs(data))
    if truncation_ignore_observables is not None:
        trunc_dist_obs_mask = ~np.isin(observable_names, truncation_ignore_observables)
        print(y_ref, trunc_dist_obs_mask)
        y_ref = y_ref[trunc_dist_obs_mask]
else:
    y_ref = np.array(y_ref)

y_lower = []
for i, obs in enumerate(observable_names):
    if trunc_dist_obs_mask[i]:
        y_lower.append(hyp["y_lower"][obs])
y_lower = np.asarray(y_lower)
print(y_lower)

trunc_dist = TruncationDistribution(
    df_0=hyp["cbar"]["degrees_of_freedom"],
    scale_0=hyp["cbar"]["scale"],
    a_0=hyp["Q"]["a"],
    b_0=hyp["Q"]["b"],
    y_lower=y_lower,
    orders_lower=np.atleast_1d(hyp["orders_lower"]),
    ignore_orders=hyp["ignore_orders"],
    y_ref=y_ref,
    update_prior=True,
    deg_quadrature=hyp["deg_quadrature"]
)

print(cov_post_lecs)

# This will do most of the work behind the scenes
obs_dist = ObservableDistribution(
    data=data,
    observables=observables,
    order=hyp["EFT_order"],
    cov_expt=cov_expt,
    mean_posterior_lecs=loc_post_lecs,
    cov_posterior_lecs=cov_post_lecs,
    lecs_prior_std=hyp["prior_standard_deviation"],
    trunc_dist=trunc_dist,
    trunc_ignore_observables=truncation_ignore_observables,
    cbar=hyp["cbar"]["fixed_value"],
    Q=hyp["Q"]["fixed_value"],
    return_predictions=mcmc["store_predictions"],
    output_file=output_file,
    observable_names=observable_names,
    parameter_names=mcmc["parameter_names"],
    compute_gradients=hyp["use_gradients"],
    posterior_lecs_indices=posterior_lecs_indices,
)

# print(obs_dist([0.89737234, 0.33464561, -0.01, -0.19]))


def log_prob_fn(p):
    # A hacky way to get MPI to play nice with a callable object
    # Must be defined before creating the pool
    # https://github.com/dfm/emcee/issues/199
    return obs_dist(p)


# We must define the pool *after* the log_prob_fn
# Otherwise MPI will complain
# If the --mpi flag is passed, then this will
# return an MPIPool object, otherwise it will
# be a SerialPool object. This helps us test the
# script before going parallel.
pool = schwimmbad.choose_pool(mpi=args.mpi)


####################################################################
# Set up the sampler and run it!
####################################################################
with pool:
    if not pool.is_master():
        # Do not repeat the actions on each child process.
        # Let the main process distribute the jobs.
        pool.wait()
        sys.exit(0)

    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Put the parameters file with the output so we know how it was generated.
    # shutil.copy2(parameters_file, output_directory)
    with open(join(output_directory, param_basename+".yaml"), 'w') as yaml_file:
        yaml.dump(input_parameters, yaml_file)

    ndim = obs_dist.n_lecs + len(obs_dist.hyper_indices)
    blobs_dtype = None
    if obs_dist.return_predictions:
        blobs_dtype = [(name, float) for name in obs_dist.observable_names]

    backend = None
    if obs_dist.output_file is not None:
        backend = emcee.backends.HDFBackend(obs_dist.output_file)

    # Do not start sampling if the file writing will just break later.
    # It will save time and prevent stupid mistakes.
    if backend is not None:
        try:
            with backend.open("a"):
                pass
        except OSError as e:
            raise OSError(
                "Make sure the backend file is closed before running the sampler"
            ) from e

    sampler = emcee.EnsembleSampler(
        nwalkers=mcmc["number_of_walkers"],
        ndim=ndim,
        log_prob_fn=log_prob_fn,
        pool=pool,
        blobs_dtype=blobs_dtype,
        backend=backend,
    )

    rng = check_random_state(mcmc["seed"])
    p0 = rng.rand(mcmc["number_of_walkers"], ndim)
    # if use_post_parameters:
    #     post_lecs_idx_with_hyperparameters = np.isin(obs_dist.parameter_names_with_hyperparameters, post_parameter_names)
    #     print(post_lecs_idx_with_hyperparameters)
    #     p0[:, post_lecs_idx_with_hyperparameters] = loc_post_lecs + 0.5 * np.sqrt(
    #         np.diag(cov_post_lecs)
    #     ) * rng.randn(mcmc["number_of_walkers"], len(posterior_lecs_indices))
    if use_post_parameters:
        post_lecs_idx_with_hyperparameters = np.isin(obs_dist.parameter_names_with_hyperparameters, post_parameter_names)
        print(post_lecs_idx_with_hyperparameters)
        p0[:, post_lecs_idx_with_hyperparameters] = loc_post_lecs + 0.5 * np.sqrt(
            np.diag(cov_post_lecs)
        ) * rng.randn(mcmc["number_of_walkers"], len(posterior_lecs_indices))
    sampler.random_state = mcmc["seed"]  # Try setting the state for reproducibility

    print("Running burn-in", flush=True)
    start_time = time.time()
    state = sampler.run_mcmc(p0, mcmc["number_of_burn_in"], store=False, progress=True)

    sampler.reset()
    # run a lot of samples
    print()
    print("Running sampler", flush=True)
    sampler.run_mcmc(state, mcmc["number_of_samples"], progress=True)

    with h5py.File(output_file, "a") as file:
        dt = h5py.special_dtype(vlen=str)
        file["mcmc"]["parameter_names"] = np.array(obs_dist.parameter_names, dtype=dt)
        file["mcmc"]["posterior_parameter_names"] = np.array(
            post_parameter_names, dtype=dt
        )
        file["mcmc"]["parameter_names_with_hyperparameters"] = np.array(
            obs_dist.parameter_names_with_hyperparameters, dtype=dt
        )

end_time = time.time()
print()
print(f"Sampling took {end_time - start_time:.1f} seconds")
print("Check out the results in", output_file)
