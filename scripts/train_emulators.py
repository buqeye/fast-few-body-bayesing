from ruamel.yaml import YAML
import schwimmbad
from argparse import ArgumentParser

import numpy as np
import pyDOE
import pickle
import os
from os.path import join, basename, splitext

from fit3bf.utils import adjust
from fit3bf import Hamiltonian
from fit3bf.observables import MultiHamiltonianTrainer
from fit3bf.utils import unpack_h5_matrices, dict_hash
from fit3bf.observables import to_hdf5, OBSERVABLE_MAP

import tqdm
import sys
import shutil
import time
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

####################################################################
# Some helper functions to make everything simpler later
####################################################################
# OPERATOR_MAP = {
#     "TritonHalfLifeOperator": TritonHalfLifeOperator,
#     "RadiusOperator": RadiusOperator,
# }


def create_hamiltonian(
    name,
    matrix_file,
    use_gradients,
    gradient_precision,
    linear_names,
    gradient_names,
):
    H0, H1, dH, lin_names, grad_names = unpack_h5_matrices(matrix_file, use_gradients)
    if np.any(lin_names != linear_names):
        raise ValueError(
            f"linear_names must match what is in h5 file for {name} Hamiltonian. "
            f"You provided {linear_names} but the Hamiltonian expects {lin_names}."
        )
    if np.any(grad_names != gradient_names) and use_gradients:
        raise ValueError(
            f"gradient_names must match what is in h5 file for {name} Hamiltonian. "
            f"You provided {gradient_names} but the Hamiltonian expects {grad_names}."
        )
    H = Hamiltonian(name, H0=H0, H1=H1, dH=dH, gradient_precision=gradient_precision)
    # MPI needs to pickle the entire object.
    # Turn this off later to save space
    H.pickle_large_attributes = True
    return H


def setup_operator(
    operator,
    matrix_file,
    use_gradients,
    linear_names,
    gradient_names,
    output_file,
    ham,
    ham_right=None,
):
    op0, op1, dop, lin_names, grad_names = unpack_h5_matrices(
        matrix_file, use_gradients
    )
    if op1 is not None and np.any(lin_names != linear_names):
        raise ValueError(
            f"linear_names must match what is in h5 file for the {name} operator. "
            f"You provided {linear_names} but the operator expects {lin_names}."
        )
    if use_gradients and np.any(grad_names != gradient_names):
        raise ValueError(
            f"gradient_names must match what is in h5 file for the {name} operator. "
            f"You provided {gradient_names} but the operator expects {grad_names}."
        )

    op = OBSERVABLE_MAP[operator](
        ham=ham, op0=op0, op1=op1, dop=dop, ham_right=ham_right
    )
    if ham.p_valid is not None:
        print("Computing validation predictions for", operator, flush=FLUSH)
        op.compute_expectation_value_validation()
    # if ham.wf_p_kernel_ is not None:
    #     op.setup_uncertainties()

    # with open(output_file, "wb") as file:
    #     pickle.dump(op, file)
    to_hdf5(op, output_file)
    return op


####################################################################
# Parse arguments, e.g.,
# python train_emulators.py --parameters=parameters/small_files.yaml
# or
# mpiexec -n 8 python train_emulators.py --parameters=parameters/small_files.yaml --mpi
####################################################################
parser = ArgumentParser(
    description="EVC emulator training script. Can train multiple Hamiltonians and operators"
)

parser.add_argument(
    "--parameters",
    dest="parameters_file",
    type=str,
    help="The parameters.yaml file that holds all input information.",
)
parser.add_argument(
    "--mpi", dest="mpi", default=False, action="store_true", help="Run with MPI."
)
args = parser.parse_args()
parameters_file = args.parameters_file
use_mpi = args.mpi

yaml = YAML(typ="safe")
with open(parameters_file, "r") as input_file:
    input_info = yaml.load(input_file)
input_hash = dict_hash(input_info)
param_basename = splitext(basename(parameters_file))[0]

# Everything is loaded. Unpack the parameters. Start the clock.
FLUSH = True  # Make "print"s happen instantaneously
start_time = time.time()

parameter_info = input_info["parameters"]
post_parameter_info = parameter_info["posterior_parameters"]
prior_parameter_names = parameter_info["names"]
posterior_parameter_names = post_parameter_info["names"]

n_train = parameter_info["number_evc_basis"]
ranges = parameter_info["ranges"]

gradient_precision = post_parameter_info["gradient_precision"]
training_params = input_info["training"]
output_directory = training_params["output_directory"]
output_directory = join(output_directory, param_basename, input_hash)
matrix_directory = training_params["matrix_directory"]

hamiltonian_params = input_info["hamiltonian"]
operator_params = input_info["operator"]

include_post_in_emulator = post_parameter_info["include_in_emulator"]
use_grads = include_post_in_emulator == "gradient"
sample_posterior_parameters = include_post_in_emulator and not use_grads


n_lecs = len(prior_parameter_names)
names = prior_parameter_names
if include_post_in_emulator and not use_grads:
    n_lecs += len(posterior_parameter_names)
    names = posterior_parameter_names + names
prior_params_mask = np.isin(names, prior_parameter_names)
post_params_mask = ~prior_params_mask

print(names)

# Create random training points
np.random.seed(parameter_info["seed"])
lhs_train = pyDOE.lhs(n_lecs, samples=n_train)
p_train = lhs_train.copy()

np.random.seed(parameter_info["validation"]["seed"])
lhs_valid = pyDOE.lhs(n_lecs, samples=parameter_info["validation"]["number"])
p_valid = lhs_valid.copy()

jj = 0
for i in range(n_lecs):
    if prior_params_mask[i]:
        p_train[:, i] = adjust(lhs_train[:, i], *ranges[jj])
        p_valid[:, i] = adjust(lhs_valid[:, i], *ranges[jj])
        jj += 1
    else:
        p_train[:, i] = adjust(lhs_train[:, i], -1, 1)
        p_valid[:, i] = adjust(lhs_valid[:, i], -1, 1)

loc_post_lecs = None
cov_post_lecs = None
ls_bounds = []
if sample_posterior_parameters:
    with open(post_parameter_info["mean_file"], "r") as input_file:
        post_param_map = yaml.load(input_file)
        loc_post_lecs = np.array(
            [post_param_map[name] for name in posterior_parameter_names]
        )
    cov_post_lecs = np.load(post_parameter_info["covariance_file"], allow_pickle=False)
    chol_post_lecs = np.linalg.cholesky(cov_post_lecs)
    n_post_stdvs = post_parameter_info["training_range_in_std_devs"]
    p_train[:, post_params_mask] = (
        loc_post_lecs
        + n_post_stdvs * (chol_post_lecs @ p_train[:, post_params_mask].T).T
    )
    p_valid[:, post_params_mask] = (
        loc_post_lecs
        + n_post_stdvs * (chol_post_lecs @ p_valid[:, post_params_mask].T).T
    )
    std_post_lecs = np.sqrt(np.diag(cov_post_lecs))
    ls_bounds = [
        (std_post_lecs[i] / 10, 3 * std_post_lecs[i]) for i in range(len(std_post_lecs))
    ]

# ls_bounds = ls_bounds + [(0.1, 5)] * len(prior_parameter_names)
# wf_kern = ConstantKernel(constant_value_bounds=(1e-5, 10)) * RBF(
#     length_scale=np.ones(n_lecs, dtype=float), length_scale_bounds=ls_bounds
# ) + WhiteKernel(1e-10, noise_level_bounds="fixed")

####################################################################
# Create an object that can distribute the eigenvector continuation
# training across multiple cores
####################################################################
print("Loading Hamiltonians", flush=FLUSH)
hamiltonians = {}
for ham_str, ham_val in hamiltonian_params.items():
    hamiltonians[ham_str] = create_hamiltonian(
        name=ham_str,
        matrix_file=join(matrix_directory, ham_val["matrix_file"]),
        use_gradients=use_grads,
        gradient_precision=gradient_precision,
        linear_names=names,
        gradient_names=posterior_parameter_names,
        # wf_p_kernel=wf_kern
    )
ham_trainer = MultiHamiltonianTrainer(**hamiltonians)
ham_trainer.make_empty_evc_containers(p_train)
ham_trainer.make_empty_evc_validation_containers(p_valid)

# The MultiHamiltonianTrainer can take each task below
# and run it in its own process. That is, it can distribute
# across both training points and Hamiltonians.
# For serial jobs, this will make no difference.
tasks = [(name, i, p, True) for name in hamiltonians for i, p in enumerate(p_train)]
tasks_validation = [
    (name, i, p, False) for name in hamiltonians for i, p in enumerate(p_valid)
]


def run_evc(task):
    # A hacky way to get MPI to play nice with a callable object
    # Must be defined before creating the pool
    # https://github.com/dfm/emcee/issues/199
    return ham_trainer(task)


pool = schwimmbad.choose_pool(mpi=use_mpi)

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
    shutil.copy2(parameters_file, output_directory)
    print("Starting tasks", flush=FLUSH)
    with tqdm.tqdm(total=len(tasks), desc="Train EVC", postfix=tasks[0][:2]) as t:

        def run_evc_callback(out):
            """The progress bar will show (hamiltonian_name, training_pt_number) at each step"""
            t.postfix = out[:2]
            t.update()

        for name, i, p_i, E, psi, dE, d_psi in pool.map(
            run_evc, tasks, callback=run_evc_callback
        ):
            # Save the results on the main process
            ham_trainer.save_evc_results(name, i, p_i, E, psi, dE, d_psi)

    print(
        f"\nThe EVC training took {time.time()-start_time:.1f} seconds to complete",
        flush=FLUSH,
    )

    for name, hamiltonian in hamiltonians.items():
        print("Storing training results for", name, flush=FLUSH)
        hamiltonian.store_fit_results()
        if training_params["estimate_evc_error"]:
            print("Training GP on wavefunction residuals")
            leave_k_out = np.array([1, 2, 3, 4])
            hamiltonian.fit_kernel_cv(
                leave_k_out, alpha=1e-7, n_restarts_optimizer=8, random_state=10
            )

    if parameter_info["validation"]["use"]:
        print("Starting validation tasks", flush=FLUSH)
        with tqdm.tqdm(
            total=len(tasks_validation),
            desc="Validate EVC",
            postfix=tasks_validation[0][:2],
        ) as t:

            def run_evc_callback(out):
                """The progress bar will show (hamiltonian_name, training_pt_number) at each step"""
                t.postfix = out[:2]
                t.update()

            for name, i, p_i, E, psi, dE, d_psi in pool.map(
                run_evc, tasks_validation, callback=run_evc_callback
            ):
                # Save the results on the main process
                E_approx = ham_trainer.hamiltonians[name].compute_gs_ritz_value(p_i)
                ritz_residual = ham_trainer.hamiltonians[name].compute_residual_vector(
                    H=ham_trainer.hamiltonians[name].compute_full_hamiltonian(p_i),
                    psi=psi,
                )
                ritz_residual_magnitude = np.sqrt(ritz_residual @ ritz_residual)
                ham_trainer.save_evc_validation_results(
                    name, i, p_i, E, psi, E_approx, ritz_residual_magnitude
                )

        print(
            f"\nThe EVC validation took {time.time()-start_time:.1f} seconds to complete",
            flush=FLUSH,
        )

    # The rest will run serially. I do not think it is worth the effort
    # to parallelize the operators.

    for name, hamiltonian in hamiltonians.items():
        hamiltonian.free_up_full()
        hamiltonian.pickle_large_attributes = False

    print("Making Operators", flush=FLUSH)
    operators = {}
    for op_str, op_val in operator_params.items():
        print(op_str)
        hamiltonian_right = None
        if op_val["hamiltonian_right"] is not None:
            hamiltonian_right = hamiltonians[op_val["hamiltonian_right"]]
        operators[op_str] = setup_operator(
            operator=op_val["operator"],
            matrix_file=join(matrix_directory, op_val["matrix_file"]),
            use_gradients=use_grads,
            linear_names=names,
            gradient_names=posterior_parameter_names,
            # output_file=join(output_directory, f"{op_str}.pickle"),
            output_file=join(output_directory, f"{op_str}.h5"),
            ham=hamiltonians[op_val["hamiltonian"]],
            ham_right=hamiltonian_right,
        )

    for name, hamiltonian in hamiltonians.items():
        to_hdf5(hamiltonian, join(output_directory, f"{name}.h5"))
        # with open(join(output_directory, f"{name}.pickle"), "wb") as ham_file:
        #     pickle.dump(hamiltonian, ham_file)

    print("Done")
    end_time = time.time()
    print(
        f"This script took {end_time-start_time:.1f} seconds to complete", flush=FLUSH
    )
    print("Check out the results in", output_directory, flush=FLUSH)
