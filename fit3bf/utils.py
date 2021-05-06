import pyDOE as pyDOE
import pickle as pickle
import numpy as np
from h5py import File
from os.path import join, splitext, basename
from re import search

from typing import Dict, Any
import hashlib
import json
import h5py
from scipy import stats


def lhsnorm(mean, cov, n, smooth=False, random_state=None):
    R"""Create latin hypercube samples from a multivariate Gaussian distribution


    THE MATLAB IMPLEMENTATION
    https://www.mathworks.com/help/stats/lhsnorm.html
    https://stackoverflow.com/a/22229362/11120345
    -------------------------

    % Generate a random sample with a specified distribution and
    % correlation structure -- in this case multivariate normal
    z = mvnrnd(mu,sigma,n);

    % Find the ranks of each column
    p = length(mu);
    x = zeros(size(z),class(z));
    for i=1:p
       x(:,i) = rank(z(:,i));
    end

    % Get gridded or smoothed-out values on the unit interval
    if (nargin<4) || isequal(dosmooth,'on')
       x = x - rand(size(x));
    else
       x = x - 0.5;
    end
    x = x / n;

    % Transform each column back to the desired marginal distribution,
    % maintaining the ranks (and therefore rank correlations) from the
    % original random sample
    for i=1:p
       x(:,i) = norminv(x(:,i),mu(i), sqrt(sigma(i,i)));
    end
    X = x;

    % -----------------------
    function r=rank(x)

    % Similar to tiedrank, but no adjustment for ties here
    [sx, rowidx] = sort(x);
    r(rowidx) = 1:length(x);
    r = r(:);
    """
    z = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=n, random_state=random_state)
    p = len(mean)
    x = np.zeros(z.shape)
    for i in range(p):
        x[:, i] = stats.rankdata(z[:, i])

    if smooth:
        x = x - np.random.rand(n)[:, None]
    else:
        x = x - 0.5
    x = x / n

    for i in range(p):
        x[:, i] = stats.norm(loc=mean[i], scale=np.sqrt(cov[i, i])).ppf(x[:, i])
    return x


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    d_hash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    d_hash.update(encoded)
    return d_hash.hexdigest()


def pack_as_h5(
    filename,
    constant,
    linear=None,
    linear_names=None,
    gradient=None,
    gradient_names=None,
):
    with h5py.File(filename, "w") as file:
        file["constant"] = constant
        if linear is not None:
            file["linear"] = linear
            if linear_names is not None:
                file.create_dataset(
                    "linear_names", data=np.asarray(linear_names, dtype="S10")
                )
        if gradient is not None:
            file["gradient"] = gradient
            if gradient_names is not None:
                file.create_dataset(
                    "gradient_names", data=np.asarray(gradient_names, dtype="S10")
                )


def unpack_h5_matrices(filename, use_gradients):
    with h5py.File(filename, "r") as file:
        op0 = file["constant"][:]
        try:
            op1 = file["linear"][:]
        except KeyError:
            op1 = None
        try:
            linear_names = np.char.decode(file["linear_names"][:], 'UTF-8')
        except KeyError:
            linear_names = None
        grad_op = None
        gradient_names = None
        if use_gradients:
            try:
                grad_op = file["gradient"][:]
            except KeyError:
                pass
            try:
                gradient_names = np.char.decode(file["gradient_names"][:], 'UTF-8')
            except KeyError:
                pass
    return op0, op1, grad_op, linear_names, gradient_names


class InputData:
    def __init__(self, path, int_name="NNLOsat", varied_lecs="3bfs"):
        if int_name == "NNLOsat":
            if varied_lecs == "3bfs":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLOsat_Nmax40_hw36_vary_3bfs.h5"),
                    "H_He3": join(path, "H_He3_NNLOsat_Nmax40_hw36_vary_3bfs.h5"),
                    "H_He4": join(path, "H_He4_NNLOsat_Nmax16_hw36_vary_3bfs.h5"),
                    "r2_He4": join(path, "r2_He4_Nmax16_hw36.txt"),
                    "E1A": join(path, "E1A_cut450_Nmax40_hw36_NNLOsat_vary_all.h5"),
                }
            elif varied_lecs == "nn-and-3bfs":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLOsat_Nmax40_hw36_vary_nn_and_3bfs.h5"),
                    "H_He3": join(
                        path, "H_He3_NNLOsat_Nmax40_hw36_vary_nn_and_3bfs.h5"
                    ),
                    "H_He4": join(
                        path, "H_He4_NNLOsat_Nmax16_hw36_vary_nn_and_3bfs.h5"
                    ),
                    "r2_He4": join(path, "r2_He4_Nmax16_hw36.txt"),
                    "E1A": join(
                        path, "E1A_cut450_Nmax40_hw36_NNLOsat_vary_nn-and-3bfs.h5"
                    ),
                }
            elif varied_lecs == "all":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLOsat_Nmax40_hw36_vary_all.h5"),
                    "H_He3": join(path, "H_He3_NNLOsat_Nmax40_hw36_vary_all.h5"),
                    "H_He4": join(path, "H_He4_NNLOsat_Nmax16_hw36_vary_all.h5"),
                    "r2_He4": join(path, "r2_He4_Nmax16_hw36.txt"),
                    "E1A": join(path, "E1A_cut450_Nmax40_hw36_NNLOsat_vary_all.h5"),
                }
            else:
                raise ValueError(
                    "That interaction",
                    int_name,
                    "and lec combination",
                    varied_lecs,
                    "is not known",
                )
        elif int_name == "NNLO_450":
            if varied_lecs == "3bfs":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLO_450_Nmax40_hw36_vary_3bfs.h5"),
                    "H_He3": join(path, "H_He3_NNLO_450_Nmax40_hw36_vary_3bfs.h5"),
                    "H_He4": join(path, "H_He4_NNLO_450_Nmax18_hw36_vary_3bfs.h5"),
                    "r2_He4": join(path, "r2_He4_Nmax18_hw36.txt"),
                    "E1A": join(path, "E1A_cut450_Nmax40_hw36_NNLO_450_vary_3bfs.h5"),
                }
            elif varied_lecs == "nn-and-3bfs":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLO_450_Nmax40_hw36_vary_nn-and-3bfs.h5"),
                    "H_He3": join(
                        path, "H_He3_NNLO_450_Nmax40_hw36_vary_nn-and-3bfs.h5"
                    ),
                    "H_He4": join(
                        path, "H_He4_NNLO_450_Nmax18_hw36_vary_nn-and-3bfs.h5"
                    ),
                    "r2_He4": join(path, "r2_He4_Nmax18_hw36.txt"),
                    "E1A": join(
                        path, "E1A_cut450_Nmax40_hw36_NNLO_450_vary_nn-and-3bfs.h5"
                    ),
                }
            elif varied_lecs == "all":
                self.path_dict = {
                    "H_H3": join(path, "H_H3_NNLO_450_Nmax40_hw36_vary_all.h5"),
                    "H_He3": join(path, "H_He3_NNLO_450_Nmax40_hw36_vary_all.h5"),
                    "H_He4": join(path, "H_He4_NNLO_450_Nmax18_hw36_vary_all.h5"),
                    "r2_He4": join(path, "r2_He4_Nmax18_hw36.txt"),
                    "E1A": join(path, "E1A_cut450_Nmax40_hw36_NNLO_450_vary_all.h5"),
                }
            else:
                raise ValueError(
                    "That interaction",
                    int_name,
                    "and lec combination",
                    varied_lecs,
                    "is not known",
                )
        else:
            raise ValueError("That interaction", int_name, "is not known")

    def verify_name(self, name):
        if name not in self.path_dict:
            raise ValueError("That operator name", name, "is not known")

    def load_varied_lecs_from_hamiltonian(self, name):
        self.verify_name(name)
        if splitext(self.path_dict[name])[1] == ".h5":
            with File(self.path_dict[name], "r") as file:
                varied_lecs = file["varied_lecs"][:].astype("<U10")
            return varied_lecs

    def load_r2_operator(self, name):
        self.verify_name(name)
        if splitext(self.path_dict[name])[1] == ".txt":
            return np.loadtxt(self.path_dict[name])

    def load_hamiltonian(self, name):
        self.verify_name(name)
        if splitext(self.path_dict[name])[1] == ".h5":
            with File(self.path_dict[name], "r") as file:
                H0 = file["H0"][:]
                H1 = file["H1"][:]
            return H0, H1

    def display_operator_info(self, name):
        self.verify_name(name)
        with open(splitext(self.path_dict[name])[0] + ".pickle", "rb") as file:
            info = pickle.load(file)
        for key in info:
            print(key)
            print(info[key])

    def load_transition(self, name):
        self.verify_name(name)
        if splitext(self.path_dict[name])[1] == ".h5":
            with File(self.path_dict[name], "r") as file:
                Op0 = file["Op0"][:]
                Op1 = file["Op1"][:]
            return Op0, Op1


class InputDataOld:
    def __init__(self, path, int_name="NNLOsat"):
        if int_name == "NNLOsat":
            H_H3_name = join(path, "H_H3_NNLOsat_Nmax40_hw36_vary_3bfs.h5")
            H_He3_name = join(path, "H_He3_NNLOsat_Nmax40_hw36_vary_3bfs.h5")
            H_He4_name = join(path, "H_He4_NNLOsat_Nmax16_hw36_vary_3bfs.h5")
            r2_He4_name = join(path, "r2_He4_Nmax16_hw36.txt")
            E1A_name = join(path, "E1A_cut450_Nmax40_hw36_vary_3bfs.h5")

        self.r2_He4 = np.loadtxt(r2_He4_name)

        with File(H_H3_name, "r") as file:
            self.H0_H3 = file["H0"][:]
            self.H1_H3 = file["H1"][:]
            self.varied_lecs = file["varied_lecs"][:].astype("<U10")

        with File(H_He3_name, "r") as file:
            self.H0_He3 = file["H0"][:]
            self.H1_He3 = file["H1"][:]

        with File(H_He4_name, "r") as file:
            self.H0_He4 = file["H0"][:]
            self.H1_He4 = file["H1"][:]

        with File(E1A_name, "r") as file:
            self.E1A_0 = file["Op0"][:]
            self.E1A_1 = file["Op1"][:]


def get_emulator_training_info(filename):
    """Extract training information and setup from emulators being used
    """
    seed_found, ntr_found = None, None
    ret_str = ""
    # search for seed
    match = search(r"seed_[0-9]*", filename)
    if match:
        seed_found = match.group()
        ret_str += seed_found
    # search for number of training points
    match = search(r"ntr_[0-9]*", filename)
    if match:
        match_found = match.group()
        ret_str += "_" + match_found

    return ret_str


def extract_file_observable_info(filename):
    """
    Extract information about which observables are included
    from a filename and returns a list

    example filename: post_mcmc_cD_cE_He4_E_He4_rptp_H3_E_T_and_D_err.npy
    """
    # all possible observables
    all_obs = ["He4_E", "He4_rptp", "H3_E", "fT12"]
    obs_found = []

    # search for these in the filename
    for obs in all_obs:
        match = search(obs, filename)
        if match:
            obs_found.append(obs)
    if not obs_found:
        print("No matches were found in this file name")
        return None
    return obs_found


def extract_file_int_info(filename):
    all_int_choices = ["NNLOsat", "NNLO_450"]
    # search for these in the filename

    for interaction in all_int_choices:
        match = search(interaction, filename)
        if match:
            return interaction

    print("No matching interactions were found in this file name")
    return None


def extract_file_vary_lecs_info(filename):
    vary_lec_choices = ["3bfs", "nn-and-3bfs", "all"]

    for choice in vary_lec_choices:
        # be careful to avoid confusing 3bfs for nn-and-3bfs by including the underscore
        match = search(r"_" + choice, filename)
        if match:
            return choice

    print("No matching vary lecs choices were found in this file name")
    return None


def check_emulator_names(files, choice_in):
    if type(files) is not list and type(files) is str:
        files = [files]
    for f in files:
        choice = extract_file_vary_lecs_info(f)
        if choice != choice_in:
            print("The file", basename(f), "does not match the choice", choice_in)
    return


def pivoted_cholesky(M):
    """Copied from Gsum.

    Needs scipy >= 1.4.0
    """
    from scipy.linalg.lapack import get_lapack_funcs

    (compute_pc,) = get_lapack_funcs(("pstrf",), arrays=(M,))
    c, p, _, info = compute_pc(M, lower=True)
    if info > 0:
        raise np.linalg.LinAlgError("M is not positive-semidefinite")
    elif info < 0:
        raise ValueError(
            "LAPACK reported an illegal value in {}-th argument"
            'on entry to "pstrf".'.format(-info)
        )

    # Compute G where M = G @ G.T
    L = np.tril(c)
    p -= 1  # The returned indices start at 1
    p_inv = np.arange(len(p))[np.argsort(p)]
    return L[p_inv]


def adjust(x, lo, hi):
    return x * (hi - lo) + lo


def generate_lhs_samples(Nsamples, domain_dimension):
    S = pyDOE.lhs(domain_dimension, samples=Nsamples)
    return S


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def compare_to_exact(computed, exact):
    percent_error = np.abs(computed - exact) / np.abs(exact) * 100.0
    print(
        "Difference = "
        + str(np.abs(computed - exact))
        + ", percent error = "
        + str(round(percent_error, 2))
    )


def bivariate_norm(x, y, pdf):
    r"""Computes the normalization constant for a bivariate probability distribution

    Uses the trapezoid rule and assumes input of pdf values directly computed on some mesh

    Parameters
        ----------
        x : array, shape = (n_x,)
            The mesh of x values on which the pdf is evaluated
        y : array, shape = (n_y,)
            The mesh of y values on which the pdf is evaluated
        pdf : array, shape = (n_x, n_y)
            Holds the pdf values computed at each n_x, n_y pair, n_x * n_y in total

        Returns
        -------
        float
            The normalization constant for the distribution
    """

    if pdf.shape[0] != x.shape[0] or pdf.shape[1] != y.shape[0]:
        print("Error, mesh size does not match x and y")
    n_x = x.shape[0]
    n_y = y.shape[0]
    norm_integral = 0.0
    p_of_x = np.zeros(n_x)
    for i in range(0, n_x):
        for j in range(1, n_y):
            delta_y = y[j] - y[j - 1]
            p_of_x[i] += delta_y / 2.0 * (pdf[i, j] + pdf[i, j - 1])
        if i > 0:
            delta_x = x[i] - x[i - 1]
            norm_integral += delta_x / 2.0 * (p_of_x[i] + p_of_x[i - 1])
    return norm_integral


def bivariate_mean(x, y, pdf):
    r"""Computes the mean point (x,y) for a bivariate probability distribution

    Uses the trapezoid rule and assumes input of pdf values directly computed on some mesh

    Parameters
        ----------
        x : array, shape = (n_x,)
            The mesh of x values on which the pdf is evaluated
        y : array, shape = (n_y,)
            The mesh of y values on which the pdf is evaluated
        pdf : array, shape = (n_x, n_y)
            Holds the pdf values computed at each n_x, n_y pair, n_x * n_y in total

        Returns
        -------
        float, float
            mean of x variable, mean of y variable
    """

    if pdf.shape[0] != x.shape[0] or pdf.shape[1] != y.shape[0]:
        print("Error, mesh size does not match x and y")
    n_x = x.shape[0]
    n_y = y.shape[0]
    mean_int_x, mean_int_y = 0.0, 0.0
    p_of_x, p_of_y = np.zeros(n_x), np.zeros(n_y)
    for i in range(0, n_x):
        for j in range(1, n_y):
            delta_y = y[j] - y[j - 1]
            p_of_x[i] += delta_y / 2.0 * (pdf[i, j] + pdf[i, j - 1])
        if i > 0:
            delta_x = x[i] - x[i - 1]
            mean_int_x += delta_x / 2.0 * (x[i] * p_of_x[i] + x[i - 1] * p_of_x[i - 1])

    for j in range(0, n_y):
        for i in range(1, n_x):
            delta_x = x[i] - x[i - 1]
            p_of_y[j] += delta_x / 2.0 * (pdf[i, j] + pdf[i - 1, j])
        if j > 0:
            delta_y = y[j] - y[j - 1]
            mean_int_y += delta_y / 2.0 * (y[j] * p_of_y[j] + y[j - 1] * p_of_y[j - 1])

    return mean_int_x, mean_int_y


def bivariate_variance(x, mu_x, y, mu_y, pdf):
    r"""Computes the normalization constant for a bivariate probability distribution

    Uses the trapezoid rule and assumes input of pdf values directly computed on some mesh

    Parameters
        ----------
        x : array, shape = (n_x,)
            The mesh of x values on which the pdf is evaluated
        mu_x : float
            The mean of the pdf in the x direction
        y : array, shape = (n_y,)
            The mesh of y values on which the pdf is evaluated
        mu_y : float
            The mean of the pdf in the y direction
        pdf : array, shape = (n_x, n_y)
            Holds the pdf values computed at each n_x, n_y pair, n_x * n_y in total

        Returns
        -------
        float, float
            mean of x variable, mean of y variable
    """

    if pdf.shape[0] != x.shape[0] or pdf.shape[1] != y.shape[0]:
        print("Error, mesh size does not match x and y")
    n_x = x.shape[0]
    n_y = y.shape[0]
    var_int_x, var_int_y = 0.0, 0.0
    p_of_x, p_of_y = np.zeros(n_x), np.zeros(n_y)
    for i in range(0, n_x):
        for j in range(1, n_y):
            delta_y = y[j] - y[j - 1]
            p_of_x[i] += delta_y / 2.0 * (pdf[i, j] + pdf[i, j - 1])
        if i > 0:
            delta_x = x[i] - x[i - 1]
            var_int_x += (
                delta_x
                / 2.0
                * (
                    (x[i] - mu_x) ** 2.0 * p_of_x[i]
                    + (x[i - 1] - mu_x) ** 2.0 * p_of_x[i - 1]
                )
            )

    for j in range(0, n_y):
        for i in range(1, n_x):
            delta_x = x[i] - x[i - 1]
            p_of_y[j] += delta_x / 2.0 * (pdf[i, j] + pdf[i - 1, j])
        if j > 0:
            delta_y = y[j] - y[j - 1]
            var_int_y += (
                delta_y
                / 2.0
                * (
                    (y[j] - mu_y) ** 2.0 * p_of_y[j]
                    + (y[j - 1] - mu_y) ** 2.0 * p_of_y[j - 1]
                )
            )

    return var_int_x, var_int_y


def find_contour_levels(grid, levels=np.array([0.68, 0.95, 0.997])):
    """
        Compute 1, 2, 3-sigma contour levels (the highest posterior density HPD) for a gridded 2D posterior
       Note: taken from BayesianAstronomy but may not work here.
    """
    sorted_ = np.sort(grid.ravel())[::-1]
    pct = np.cumsum(sorted_) / np.sum(sorted_)
    cutoffs = np.searchsorted(pct, levels)
    return np.sort(sorted_[cutoffs])


def covariance(x, mu_x, y, mu_y, pdf):
    r"""Computes the covariance between two variables assuming a bivariate distribution

    Uses the trapezoid rule and assumes input of pdf values directly computed on some mesh

    Parameters
        ----------
        x : array, shape = (n_x,)
            The mesh of x values on which the pdf is evaluated
        mu_x : float
            The mean of the pdf in the x direction
        y : array, shape = (n_y,)
            The mesh of y values on which the pdf is evaluated
        mu_y : float
            The mean of the pdf in the y direction
        pdf : array, shape = (n_x, n_y)
            Holds the pdf values computed at each n_x, n_y pair, n_x * n_y in total

        Returns
        -------
        float
            covariance sigma_xy
    """
    if pdf.shape[0] != x.shape[0] or pdf.shape[1] != y.shape[0]:
        print("Error, mesh size does not match x and y")
    n_x = x.shape[0]
    n_y = y.shape[0]
    cov_int = 0
    p_of_x = np.zeros(n_x)
    for i in range(0, n_x):
        for j in range(1, n_y):
            delta_y = y[j] - y[j - 1]
            p_of_x[i] += (
                delta_y
                / 2.0
                * ((y[j] - mu_y) * pdf[i, j] + (y[j - 1] - mu_y) * pdf[i, j - 1])
            )
        if i > 0:
            delta_x = x[i] - x[i - 1]
            cov_int += (
                delta_x
                / 2.0
                * ((x[i] - mu_x) * p_of_x[i] + (x[i - 1] - mu_x) * p_of_x[i - 1])
            )
    return cov_int
