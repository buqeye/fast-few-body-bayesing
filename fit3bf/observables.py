import numpy as np
from numpy.linalg import solve
import scipy as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy import stats
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
from tqdm import tqdm
import docrep


op_docstrings = docrep.DocstringProcessor()


def verify_trained(f):
    """A simple decorator for Hamiltonian methods

    It will make them throw an error if they haven't been trained first.
    """
    def wrapper(self, *args, **kwargs):
        self.verify_evc()
        return f(self, *args, **kwargs)
    return wrapper


@op_docstrings.get_sectionsf('Hamiltonian')
@op_docstrings.dedent
class Hamiltonian:
    R"""A class that solves the eigenvalue problem for a Hamiltonian that depends linearly on parameters.

    Can setup and use eigenvector continuation (EVC) to speed up calculations, and also has methods to compare
    exact results to EVC results.
    The eigenvalues and eigenvectors found using EVC are called Ritz values and vectors, since EVC
    is essentially the Ritz method for subspace evaluation.

    Parameters
    ----------
    name : str
        The name of the system
    H0 : array, shape = (N, N), optional
        The part of the Hamiltonian that is constant with respect to the parameters
    H1 : array, shape = (N, N, n_p), optional
        The part of the Hamiltonian that depends linearly on the parameters
    """

    _large_attrs = [
        'H0',
        'H1',
        'dH',
        'X_sub',
        'dX_sub',
        'X_valid',
        'gp_'
    ]

    def __init__(self, name, H0, H1, dH=None, gradient_precision=1e-13):
        self.name = name

        self.H0 = H0
        self.H1 = H1
        self.dH = dH
        self.n_rows_H = None
        self.n_lecs = None
        self.n_grads = None
        # Subspace-related quantities for EVC emulator training
        self.H0_sub = None
        self.H1_sub = None
        self._is_evc_completed = False
        self.N_sub = None
        self.E_train = None
        self.p_train = None
        self.X_sub = None
        self.dE_train = None
        self.dN_sub = None
        self.dX_sub = None
        self.dH_sub = None

        self.p_valid = None
        self.E_valid_true = None
        self.E_valid_approx = None
        self.X_valid = None
        self.X_valid_residual_magnitude = None

        self.pickle_large_attributes = False
        self.has_gradients = False
        self.gradient_precision = gradient_precision

        self._json_attributes = [
            "wf_p_kernel",
            "wf_p_kernel_",
        ]

    def to_dict(self, with_cls_name=False):
        d = self.__dict__.copy()  # Only need shallow copy
        import json_tricks
        from ast import literal_eval
        if not self.pickle_large_attributes:
            for key in d:
                if key in self._large_attrs:
                    d[key] = None
        for key, val in d.items():
            if key in self._json_attributes and val is not None:
                # d[key] = literal_eval(json_tricks.dumps(val))
                d[key] = json_tricks.dumps(val)
        if with_cls_name:
            return {"class_name": self.__class__.__name__, "attributes": d}
        return d

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        import json_tricks

        state = state.copy()
        for key, val in state.items():
            if key in state["_json_attributes"] and val is not None:
                # state[key] = json_tricks.loads(str(val))
                state[key] = json_tricks.loads(val)
        self.__dict__.update(state)

    def _to_hdf5(self, filename, mode="w", **kwargs):
        import hickle
        d = self.to_dict(with_cls_name=True)
        with open(filename, mode=mode, **kwargs) as f:
            hickle.dump(d, f)
        return self

    @classmethod
    def from_dict(cls, state):
        self = cls.__new__(cls)
        self.__setstate__(state)
        return self

    # @classmethod
    # def _from_hdf5(cls, filename, path="/", safe=True):
    #     import hickle
    #     with open(filename, "r") as f:
    #         state: dict = hickle.load(f, path=path, safe=safe)
    #     return cls.from_dict(state)

    def verify_evc(self):
        if not self._is_evc_completed:
            raise ValueError('Cannot complete this action until EVC is performed')

    def verify_full(self):
        if (self.H0 is None) or (self.H1 is None):
            raise ValueError('Cannot complete this action because the full Hamiltonian is not available')

    def full_available(self):
        try:
            self.verify_full()
            is_available = True
        except ValueError:
            is_available = False
        return is_available

    def free_up_full(self, all_large=False):
        """Dump the full Hamiltonian information to free up memory
        """
        # self.verify_full()
        self.verify_evc()
        # let Python garbage collection take care of de-allocation
        self.H0 = None
        self.H1 = None
        if all_large:
            for attr in self._large_attrs:
                setattr(self, attr, None)

    def compute_full_hamiltonian(self, p):
        self.verify_full()
        return self.H0 + self.H1 @ p

    def compute_subspace_hamiltonian(self, p, evc_subset=None):
        self.verify_evc()
        H0, H1 = self.H0_sub, self.H1_sub
        if evc_subset is not None:
            H0 = H0[evc_subset][:, evc_subset]
            H1 = H1[evc_subset][:, evc_subset]
        return H0 + H1 @ p

    @staticmethod
    def compute_full_gs_from_ham(H):
        # return eigh(H, eigvals=(0, 0))
        E, psi = eigsh(H, k=1, which='SA', tol=1e-6)
        return E[0], psi.ravel()

    def compute_full_ground_state(self, p):
        self.verify_full()
        H = self.compute_full_hamiltonian(p)
        return self.compute_full_gs_from_ham(H)

    def solve_subspace_gen_eigenvalue_problem(self, p, evc_subset=None, **kwargs):
        R"""Computes the Ritz values and vectors by solving the generalized eigenvalue problem

        .. math::
            X^T H X \beta = E_{ritz} N \beta

        where :math:`X` and :math:`N` are precomputed by the `fit_evc` method.
        The :math:`\beta` returned are orthonormal,
        in the sense that :math:`{\beta_i}^T N \beta_j = \delta_{ij}`.
        Thus, the Ritz vectors :math:`|\psi_{ritz}\rangle = X \beta` created from these coefficients
        are also normalized such that :math:`\langle\psi_{ritz} | \psi_{ritz}\rangle = 1`.

        Parameters
        ----------
        p : array, shape = (n_p,)
            The parameter values at which to evaluate the Hamiltonian
        evc_subset : array,
            The indices of the EVC basis to use in the computation. Defaults to None, which
            uses all of them.
        kwargs : optional
            The keyword parameters passed to `eigh`.

        Returns
        -------
        E : array, shape = (num_basis,)
        beta : array, shape = (num_basis, num_basis)
        """
        H_sub = self.compute_subspace_hamiltonian(p, evc_subset=evc_subset)
        N = self.N_sub
        if evc_subset is not None:
            N = N[evc_subset][:, evc_subset]
        E_sub, beta = eigh(H_sub, b=N, type=1, **kwargs)
        return E_sub, beta

    def compute_wave_function_coefficients(self, p, **kwargs):
        R"""Return the coefficients :math:`\beta` that multiply the basis of exact wavefunctions to create Ritz vectors

        .. math::
            |\psi_{ritz}\rangle = X \beta

        Parameters
        ----------
        p : array, shape=(n_p,)
        kwargs : optional
        """
        return self.solve_subspace_gen_eigenvalue_problem(p, **kwargs)[-1]

    def compute_gs_wave_function_coefficients(self, p, **kwargs):
        return self.compute_wave_function_coefficients(p, eigvals=(0, 0), **kwargs).ravel()

    def compute_ritz_values(self, p, **kwargs):
        R"""Return the vector of Ritz values :math:`\{E_{ritz}\}_i`

        Parameters
        ----------
        p
        kwargs

        Returns
        -------

        """
        return self.solve_subspace_gen_eigenvalue_problem(p, **kwargs)[0]

    def compute_gs_ritz_value(self, p, **kwargs):
        R"""Return the smallest Ritz value :math:`E_{ritz}`"""
        return self.compute_ritz_values(p, eigvals=(0, 0), **kwargs)[0]

    def compute_ritz_vectors(self, p, evc_subset=None, **kwargs):
        R"""Return the set of Ritz vectors :math:`\{|\psi_ritz\rangle\}_i`

        Returns
        -------
        ritz : array, shape = (N, n_subspace)
        """
        _, beta = self.solve_subspace_gen_eigenvalue_problem(p, evc_subset=evc_subset, **kwargs)
        X = self.X_sub
        if evc_subset is not None:
            X = X[:, evc_subset]
        return X @ beta

    def compute_gs_ritz_vector(self, p, **kwargs):
        R"""Return the ground state of Ritz vector :math:`|\psi_ritz\rangle`

        Returns
        -------
        ritz_0 : array, shape = (N,)
        """
        return self.compute_ritz_vectors(p, eigvals=(0, 0), **kwargs)[:, 0]

    def compute_gs_ritz_value_gradient(self, p, **kwargs):
        R"""Return the smallest Ritz value :math:`E_{ritz}`"""
        beta = self.compute_gs_wave_function_coefficients(p, **kwargs)
        dE = beta.conj().T @ self.dH_sub @ beta
        return dE

    def compute_gs_wave_function_coefficients_gradients(self, p, **kwargs):
        E, beta = self.solve_subspace_gen_eigenvalue_problem(p, eigvals=(0, 0), **kwargs)
        from .gradients import grad_eigh_ritz
        _, d_beta = grad_eigh_ritz(E, beta, dH=self.dH_sub, dN=self.dN_sub)
        return d_beta[..., 0]

    def fit_evc(self, p_train):
        R"""Creates the subspace basis :math:`X` and norm :math:`N` using eigenvector continuation

        Calling this method is required before calling anything else using eigenvector continuation.

        The idea behind the approach is to approximate the exact wavefunction
        at some parameter value :math:`p` as a linear combination of exact wavefunctions
        :math:`|\psi_i\rangle` at a set of :math:`p_{i}`. That is,

        .. math::
            |\psi\rangle \approx |\psi_{ritz}\rangle = \sum \beta_i |\psi_i\rangle = X \beta

        The coefficients :math:`\beta` for any :math:`p` can then be found
        by solving the generalized eigenvector problem

        .. math::
            X^T H X \beta = E_{ritz} N \beta

        where :math:`N = X^T X`. The :math:`E_{ritz}` and :math:`|\psi_{ritz}\rangle` are known
        as the Ritz values and vectors.

        Parameters
        ----------
        p_train : array, shape = (n_basis, n_params)
            The parameters of the Hamiltonian at which to solve the full system, and append the resulting
            eigenvector to the subspace basis.

        Returns
        -------
        self
        """
        # from mpi4py.futures import MPIPoolExecutor
        self.verify_full()

        num_basis = p_train.shape[0]
        n_rows_H = self.H0.shape[0]
        X_sub = np.NaN * np.ones((n_rows_H, num_basis))
        E_sub = np.zeros(num_basis)
        dE_sub = None
        dX_sub = None
        dH = self.dH
        has_gradients = dH is not None
        if has_gradients:
            dE_sub = np.zeros((dH.shape[0], num_basis.shape[-1]), dtype="float64")
            dX_sub = np.zeros((dH.shape[0], *X_sub.shape), dtype=X_sub.dtype)

        # H0, H1 = self.H0, self.H1
        #
        # # MPI doesn't like to pickle `self`, so create the necessary func
        # def compute_full_ground_state(p):
        #     H = H0 + H1 @ p
        #     E_, psi_ = eigsh(H, k=1, which='SA', tol=1e-6)
        #     return E_[0], psi_.ravel()

        # print('Creating EVC basis...', flush=True)

        for i, p_i in tqdm(enumerate(p_train), total=len(p_train), postfix={'name': self.name}):
            E, psi = self.compute_full_ground_state(p_i)
            X_sub[:, i] = psi
            E_sub[i] = E
            if has_gradients:
                dE_sub[..., i] = dE = (psi.conj().T @ dH @ psi).real
                dX_sub[..., i] = self.compute_single_eigenvector_gradient(
                    p_i, E, dE, psi, precision=self.gradient_precision
                )

        # Must have a working version of MPI, e.g.,
        # conda install openmpi
        # with MPIPoolExecutor(max_workers=4) as executor:
        #     # results = executor.map(gs_with_count, np.arange(num_basis), p_train)
        #     results = executor.map(compute_full_ground_state, p_train)
        #     i = 0
        #     for E, psi in results:
        #         if True:
        #             print(i, flush=True)
        #         X_sub[:, i] = psi
        #         E_sub[:, i] = E
        #         i += 1
        self.p_train = p_train
        self.E_train = E_sub
        self.X_sub = X_sub
        self.dE_train = dE_sub
        self.dX_sub = dX_sub
        self.has_gradients = has_gradients
        self.store_fit_results()
        return self

    def store_fit_results(self):
        X_sub = self.X_sub
        N_sub = X_sub.T @ X_sub

        self.N_sub = N_sub
        self.H0_sub = X_sub.T @ self.H0 @ X_sub

        # These parentheses make the code much faster!
        H1_sub_reshaped = X_sub.T @ (np.transpose(self.H1, (2, 0, 1)) @ X_sub)
        self.H1_sub = np.transpose(H1_sub_reshaped, (1, 2, 0))
        # self.H1_sub = np.einsum('ij,jkp,kl->ilp', X_sub.T, self.H1, X_sub)
        self._is_evc_completed = True

        self.n_rows_H = self.H0.shape[0]
        self.n_lecs = self.H1.shape[-1]

        if self.has_gradients:
            self.n_grads = self.dH.shape[0]
            self.dH_sub = X_sub.conj().T @ self.dH @ X_sub
            self.dN_sub = self.compute_grad_norm_matrix(X_sub, self.dX_sub)
        return self

    def __call__(self, p, use_evc=True, return_gradient=False):
        R"""Computes the ground state energy using eigenvector continuation

        Parameters
        ----------
        p

        Returns
        -------
        E : float
            The ground state energy
        """
        if use_evc:
            E = self.compute_gs_ritz_value(p)
            if return_gradient:
                dE = self.compute_gs_ritz_value_gradient(p)
                return E, dE
            return E
        else:
            E = self.compute_full_ground_state(p)[0]
            return E

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'

    def compute_single_eigenvector_gradient(self, p_i, E_i, dE_i, X_i, precision, offset=None):
        from .gradients import grad_eigenvectors_hermitian_iterative
        H = self.compute_full_hamiltonian(p_i)
        if offset is None:
            # Find largest eigenvalue because we have to subtract it off
            # If we wanted to be less careful, we could instead just use a really big number
            offset = eigsh(H, k=1, which='LA', tol=1e-6, return_eigenvectors=False)
        dX_i = grad_eigenvectors_hermitian_iterative(
            H=H, dH=self.dH, val=E_i, vec=X_i, d_val=dE_i, offset=offset, verbose=False, precision=precision
        )
        return dX_i

    @staticmethod
    def compute_grad_norm_matrix(X, dX):
        return np.transpose(dX.conj(), axes=(0, 2, 1)) @ X + X.conj().T @ dX

    def compute_true_vs_approximate(self, p_samples):
        true = np.asarray([self.compute_full_ground_state(p)[0] for p in p_samples])
        approx = np.asarray([self(p) for p in p_samples])
        return true, approx

    def display_true_vs_approximate_error(self, p_samples, percent_error=False):

        # some checks to let the user input a single point or a list of points in different formats
        if type(p_samples) is not np.ndarray:
            p_samples = np.asarray(p_samples)
        if p_samples.ndim < 2:
            p_samples = np.asarray([p_samples])

        true, approx = self.compute_true_vs_approximate(p_samples)
        for idx, val in enumerate(true):
            print('-> Target LEC point', idx+1)
            print('>', self.name, 'Exact   ', ':', '{:f}'.format(val))
            print('>', self.name, 'Emulator', ':', '{:f}'.format(approx[idx]))
            print('>', self.name, '% error ', ':', '{:f}'.format(100.0 * abs((val - approx[idx])/val)), ' %')

    @staticmethod
    def compute_residual_vector(H, psi):
        R"""The residual vector :math:`r(\psi)`

        Note that this is not the residual between Ritz and exact eigenvectors.
        Rather, it is defined to be

        .. math::
            H |\psi\rangle - |\psi\rangle \frac{\langle\psi | H | \psi\rangle}{\langle\psi | \psi\rangle}

        Parameters
        ----------
        H : array, shape = (N,N)
            The Hamiltonian
        psi : array, shape = (N,)
            A vector

        Returns
        -------
        r : array, shape = (N,)
            The residual vector
        """
        return H @ psi - psi * (psi.T @ H @ psi) / (psi.T @ psi)


@op_docstrings.get_sectionsf('Operator')
@op_docstrings.dedent
class Operator:
    R"""A quantum operator that can be evaluated between two eigenstates of a Hamiltonian

    Supports expectation values between states from 1 or 2 different Hamiltonians:

    .. math::
        \langle O \rangle = \langle \psi_{left} | O | \psi_{right}\rangle

    Supports quick evaluation using eigenvector continuation. The Hamiltonians must
    have EVC subspace set up before passing to this object.

    Parameters
    ----------
    name : str
        The name of the operator
    ham : Hamiltonian
        The (left) Hamiltonian object. If `ham_right is None`, then it is assumed that this
        is also the Hamiltonian for the right state.
    op0 : array, shape = (N, N)
        The large matrix elements for this operator.
    op1 : array, shape = (N, N, n_p)
        The large matrix elements for this operator that depend linearly on parameters.
        Optional, only if the operator depends on LECs.
    ham_right : Hamiltonian
        The right Hamiltonian object. Optional, only needed if the operator is sandwiched
        between two different states.
        If given, then `ham` is the left Hamiltonian.
    """

    _large_attrs = [
        'op0',
        'op1',
        'dop',
    ]

    def __init__(self, name, ham, op0, op1=None, ham_right=None, dop=None):
        # Check that the provided inputs match the name of the desired operator to compute
        self.name = name

        self.ham_left = ham
        self.ham_right = ham_right
        self.p_train = self.ham_left.p_train
        self.n_lecs = ham.n_lecs

        # make sure all the training points stuff matches for all inputs
        self.verify_training_pts()

        if ham_right is not None:
            self._transition = True
        else:
            self._transition = False

        X_sub_left = X_sub_right = self.ham_left.X_sub
        if self._transition:
            X_sub_right = self.ham_right.X_sub

        self.op0 = op0
        self.op1 = op1
        self.dop = dop
        self.N_sub_lr = X_sub_left.T @ X_sub_right
        self.op0_sub = X_sub_left.T @ op0 @ X_sub_right
        self.op1_sub = None
        self.dXt_op0_X = None
        self.dXt_op1_X = None
        self.Xt_op0_dX = None
        self.Xt_op1_dX = None
        self.Xt_dop_X = None
        if op1 is not None:
            # self.op1_sub = np.einsum('ij,jkp,kl->ilp', X_sub_left.T, op1, X_sub_right)
            op1_t = np.transpose(op1, axes=(2, 0, 1))
            self.op1_sub = np.transpose(X_sub_left.T @ (op1_t @ X_sub_right), axes=(1, 2, 0))

        if dop is not None or self.ham_left.has_gradients:
            self.setup_gradient_matrices()

    def __getstate__(self):
        d = self.__dict__.copy()  # Only need shallow copy
        for key in d:
            if key in self._large_attrs:
                d[key] = None
        return d

    def _to_hdf5(self, filename, mode="w", **kwargs):
        import hickle
        d = {"class_name": self.__class__.__name__, "attributes": self.__getstate__()}
        state = d["attributes"]
        state["ham_left"] = state["ham_left"].to_dict(with_cls_name=True)
        if state["ham_right"] is not None:
            state["ham_right"] = state["ham_right"].to_dict(with_cls_name=True)
        with open(filename, mode=mode, **kwargs) as f:
            hickle.dump(d, f)
        return self

    @classmethod
    def from_dict(cls, state):
        self = cls.__new__(cls)
        self.__dict__ = state.copy()
        ham_left_cls = OBSERVABLE_MAP[self.__dict__["ham_left"]["class_name"]]
        self.__dict__["ham_left"] = ham_left_cls.from_dict(self.__dict__["ham_left"]["attributes"])
        if self.__dict__["ham_right"] is not None:
            ham_right_cls = OBSERVABLE_MAP[self.__dict__["ham_right"]["class_name"]]
            self.__dict__["ham_right"] = ham_right_cls.from_dict(self.__dict__["ham_right"]["attributes"])
        return self

    def setup_gradient_matrices(self):
        X_l = X = self.ham_left.X_sub
        dX_l = dX = self.ham_left.dX_sub
        if self._transition:
            X = self.ham_right.X_sub
            dX = self.ham_right.dX_sub
        dO = self.dop
        op0 = self.op0
        op1 = self.op1
        n_evc = X_l.shape[-1]
        n_grad = dX_l.shape[0]
        Xt = X_l.conj().T
        dXt = np.transpose(dX_l.conj(), axes=(0, 2, 1))

        self.dXt_op0_X = dXt @ op0 @ X
        # self.dXt_op1_X = np.zeros((n_grad, n_evc, n_evc, n_p))
        self.dXt_op1_X = None
        op1_t = None
        if op1 is not None:
            op1_t = np.transpose(op1, axes=(2, 0, 1))
            dXt_padded = dXt[:, np.newaxis, :, :]
            self.dXt_op1_X = np.transpose(dXt_padded @ (op1_t @ X), axes=(0, 2, 3, 1))
            # self.dXt_op1_X = np.einsum('nij,jkp,kl->nilp', dXt, op1, X)

        self.Xt_op0_dX = Xt @ op0 @ dX
        # self.Xt_op1_dX = np.zeros((n_grad, n_evc, n_evc, n_p))
        self.Xt_op1_dX = None
        if op1 is not None:
            dX_padded = dX[:, np.newaxis, :, :]
            self.Xt_op1_dX = np.transpose((Xt @ op1_t) @ dX_padded, axes=(0, 2, 3, 1))
            # self.Xt_op1_dX = np.einsum('ij,jkp,nkl->nilp', Xt, op1, dX)

        self.Xt_dop_X = np.zeros((n_grad, n_evc, n_evc))
        if dO is not None:
            self.Xt_dop_X = Xt @ dO @ X

    def verify_training_pts(self):
        """Checks that Hamiltonians were generated with the same p_train value
        """
        if not np.allclose(self.p_train, self.ham_left.p_train):
            raise ValueError('The Hamiltonian training points are not the same as used for the operator')
        if self.ham_right is not None:
            if not np.allclose(self.ham_left.p_train, self.ham_right.p_train):
                raise ValueError('The Hamiltonian training points (right and left) are not the same')

    def verify_full_op(self):
        if self.op0 is None:
            raise ValueError('The operator information needed for this action is not available.')

    def compute_full_operator(self, p):
        op = self.op0
        if self.op1 is not None:
            op = op + self.op1 @ p
        return op

    def compute_subspace_operator(self, p):
        op = self.op0_sub
        if self.op1_sub is not None:
            op = op + self.op1_sub @ p
        return op

    def compute_expectation_value_validation(self):
        p_valid = self.ham_left.p_valid
        if p_valid is None:
            raise ValueError('Validation can only be run if the Hamiltonians have run on validation data')
        if self._transition and not np.allclose(p_valid, self.ham_right.p_valid):
            raise ValueError("The left and right Hamiltonians were not computed on the same validation data")
        ev_valid_true = []
        ev_valid_approx = []
        for i, p_i in enumerate(p_valid):
            psi_left = psi_right = self.ham_left.X_valid[:, i]
            if self._transition:
                psi_right = self.ham_right.X_valid[:, i]
            op = self.compute_full_operator(p_i)
            ev_i = psi_left.T @ op @ psi_right
            ev_valid_true.append(ev_i)
            ev_valid_approx.append(self.expectation_value_subspace(p_i))
        self.p_valid = p_valid
        self.ev_valid_true = np.array(ev_valid_true)
        self.ev_valid_approx = np.array(ev_valid_approx)
        return self.ev_valid_true, self.ev_valid_approx

    def expectation_value_full(self, p):
        psi_left, psi_right = self.compute_full_gs_psi_left_and_right(p)
        op = self.compute_full_operator(p)
        return psi_left.T @ op @ psi_right

    def compute_full_gs_psi_left_and_right(self, p):
        _, psi_left = self.ham_left.compute_full_ground_state(p)
        if self._transition:
            _, psi_right = self.ham_right.compute_full_ground_state(p)
        else:
            psi_right = psi_left
        return psi_left, psi_right

    def compute_gs_beta_left_and_right(self, p):
        beta_left = self.ham_left.compute_gs_wave_function_coefficients(p)
        if self._transition:
            beta_right = self.ham_right.compute_gs_wave_function_coefficients(p)
        else:
            beta_right = beta_left
        return beta_left, beta_right

    def compute_gs_beta_left_and_right_gradients(self, p):
        d_beta_left = self.ham_left.compute_gs_wave_function_coefficients_gradients(p)
        if self._transition:
            d_beta_right = self.ham_right.compute_gs_wave_function_coefficients_gradients(p)
        else:
            d_beta_right = d_beta_left
        return d_beta_left, d_beta_right

    def expectation_value_subspace(self, p):
        R"""

        The beta are normalized so no normalization of this quantity
        should be necessary.

        Parameters
        ----------
        p

        Returns
        -------

        """
        beta_left, beta_right = self.compute_gs_beta_left_and_right(p)
        op = self.compute_subspace_operator(p)
        return beta_left.T @ op @ beta_right

    def expectation_value_gradient_subspace(self, p):
        beta_left, beta_right = self.compute_gs_beta_left_and_right(p)
        d_beta_left, d_beta_right = self.compute_gs_beta_left_and_right_gradients(p)
        op = self.compute_subspace_operator(p)

        left_grad_op = self.dXt_op0_X
        right_grad_op = self.Xt_op0_dX
        if self.op1 is not None:
            left_grad_op = left_grad_op + self.dXt_op1_X @ p
            right_grad_op = right_grad_op + self.Xt_op1_dX @ p

        return (
                beta_left @ left_grad_op @ beta_right +
                beta_left @ right_grad_op @ beta_right +
                d_beta_left @ op @ beta_right + (beta_left @ op @ d_beta_right.T).T +
                beta_left @ self.Xt_dop_X @ beta_right
        )

    def __call__(self, p, use_evc=True, return_std=False, return_gradient=False):
        R"""Computes the operator expectation value between states of the left and right Hamiltonian

        Uses eigenvector continuation for speed

        Parameters
        ----------
        p : array, shape = (n_p,)
            The Hamiltonian and Operator parameters at which to compute the expectation value
        use_evc : bool, optional
            Whether to use eigenvector continuation. Defaults to True.
        return_std : bool, optional
            Whether to return the standard deviation due to EVC. Experimental. Defaults to False.

        Returns
        -------
        float
            The expectation value
        """
        if use_evc:
            ev = self.expectation_value_subspace(p)
            if return_gradient:
                dev = self.expectation_value_gradient_subspace(p)
                return ev, dev
            if return_std:
                std = self.compute_std(p)
                return ev, std
            return ev
        else:
            return self.expectation_value_full(p)

    def __repr__(self):
        if self._transition:
            return f'<{self.ham_left.name} | {self.name} | {self.ham_right.name}>'
        else:
            return f'<{self.ham_left.name} | {self.name} | {self.ham_left.name}>'

    def compute_true_vs_approximate(self, p_samples):
        true = np.asarray([self(p, use_evc=False) for p in p_samples])
        approx = np.asarray([self(p, use_evc=True) for p in p_samples])
        return true, approx


op_docstrings.delete_params('Operator.parameters', 'name')


@op_docstrings.dedent
class PropagatedOperator(Operator):
    R"""A matrix element whose value is to be propagated to some other value

    This is meant to be subclassed by overriding the `propagate` method.

    Parameters
    ----------
    %(Operator.parameters.no_name)s
    """

    name = 'Op'
    propagate_str = 'Identity function'

    def __init__(self, *args, **kwargs):
        super().__init__(self.name, *args, **kwargs)

    @staticmethod
    def propagate(matrix_element, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def propagate_gradient(matrix_element, grad_matrix_element, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, p, use_evc=True, return_std=False, return_gradient=False):
        out = super().__call__(p, use_evc=use_evc, return_std=return_std, return_gradient=return_gradient)
        if return_gradient:
            out, grad_out = out
            return self.propagate(out), self.propagate_gradient(out, grad_out)
        if return_std:
            ev, ev_std = out
            dist = stats.norm(loc=ev, scale=ev_std)
            prop_ev_samples = self.propagate(dist.rvs(500))
            return self.propagate(ev), np.std(prop_ev_samples)
        return self.propagate(out)

    def compute_cov(self, p, pp=None, use_evc=True):
        cov = super().compute_cov(p=p, pp=pp, use_evc=use_evc)
        get_ev = super().__call__
        ev = np.array([
            get_ev(p_i, use_evc=use_evc, return_std=False) for p_i in p
        ])
        dist = stats.multivariate_normal(mean=ev, cov=cov)
        prop_ev_samples = self.propagate(dist.rvs(500))
        return np.cov(prop_ev_samples.T)

    def compute_expectation_value_validation(self):
        ev_valid_true, ev_valid_approx = super().compute_expectation_value_validation()
        self.ev_valid_true = self.propagate(ev_valid_true)
        self.ev_valid_approx = self.propagate(ev_valid_approx)

    def __repr__(self):
        str_super = super().__repr__()
        return f'{self.propagate_str}({str_super})'


@op_docstrings.dedent
class RadiusOperator(PropagatedOperator):
    R"""The radius operator

    The initialization still expects matrix elements for R squared, but when
    called, this object returns the square root of the expectation value.

    Parameters
    ----------
    %(Operator.parameters.no_name)s
    """

    name = 'R2'
    propagate_str = 'sqrt'

    @staticmethod
    def propagate(matrix_element, *args, **kwargs):
        return np.sqrt(matrix_element)

    @staticmethod
    def propagate_gradient(matrix_element, grad_matrix_element, *args, **kwargs):
        return 0.5 * grad_matrix_element / np.sqrt(matrix_element)


def compute_ft12(E1A):
    K_over_GV2 = 6146.6
    fA_over_fV = 1.00529
    delta_c = 0.0013
    return K_over_GV2 / (1 - delta_c + 3 * np.pi * fA_over_fV * E1A**2)


def compute_ft12_grad(E1A, dE1A):
    fA_over_fV = 1.00529
    delta_c = 0.0013
    prefactor = 6 * np.pi * fA_over_fV * E1A * dE1A
    return prefactor * compute_ft12(E1A) / (1 - delta_c + 3 * np.pi * fA_over_fV * E1A**2)


@op_docstrings.dedent
class TritonHalfLifeOperator(PropagatedOperator):
    R"""A convenience class that converts an E1A operator to a half life operator

    Parameters
    ----------
    %(Operator.parameters.no_name)s
    """

    name = 'E1A'
    propagate_str = 'fT12'

    @staticmethod
    def propagate(matrix_element, *args, **kwargs):
        return compute_ft12(matrix_element)

    @staticmethod
    def propagate_gradient(matrix_element, grad_matrix_element, *args, **kwargs):
        return compute_ft12_grad(matrix_element, grad_matrix_element)


class MultiHamiltonianTrainer:
    """This class allows MPI to distribute the calculations of multiple Hamiltonians across various cores.
    """

    def __init__(self, **hamiltonians: Hamiltonian):
        self.hamiltonians = hamiltonians

    def make_empty_evc_containers(self, p_train):
        """Warning: This will overwrite X_sub, E_sub, and p_train in each Hamiltonian"""
        num_basis = p_train.shape[0]
        for ham in self.hamiltonians.values():
            n_rows_H = ham.H0.shape[0]
            ham.X_sub = X = np.NaN * np.ones((n_rows_H, num_basis))
            ham.E_train = np.zeros(num_basis)
            ham.p_train = np.zeros(p_train.shape)

            dH = ham.dH
            ham.has_gradients = dH is not None
            if ham.has_gradients:
                ham.dE_train = np.zeros((dH.shape[0], num_basis), dtype="float64")
                ham.dX_sub = np.zeros((dH.shape[0], *X.shape), dtype=X.dtype)

    def make_empty_evc_validation_containers(self, p_valid):
        """Warning: This will overwrite X_sub, E_sub, and p_train in each Hamiltonian"""
        num_basis = p_valid.shape[0]
        for ham in self.hamiltonians.values():
            n_rows_H = ham.H0.shape[0]
            ham.X_valid = np.NaN * np.ones((n_rows_H, num_basis))
            ham.E_valid_true = np.zeros(num_basis)
            ham.E_valid_approx = np.zeros(num_basis)
            ham.p_valid = np.zeros(p_valid.shape)
            ham.X_valid_residual_magnitude = np.zeros(num_basis)

            dH = ham.dH
            ham.has_gradients = dH is not None

    def __call__(self, task):
        name, i, params, return_gradients = task
        H = self.hamiltonians[name]
        E, psi = H.compute_full_ground_state(params)
        precision = H.gradient_precision
        dE = None
        d_psi = None
        if H.dH is not None and return_gradients:
            dE = (psi.conj().T @ H.dH @ psi).real
            d_psi = H.compute_single_eigenvector_gradient(params, E, dE, psi, precision=precision)
        return name, i, params, E, psi, dE, d_psi

    def save_evc_results(self, name, i, p, E, psi, dE, d_psi):
        H = self.hamiltonians[name]
        H.E_train[i] = E
        H.p_train[i] = p
        H.X_sub[:, i] = psi
        if dE is not None and d_psi is not None:
            H.dE_train[..., i] = dE
            H.dX_sub[..., i] = d_psi
        return self

    def save_evc_validation_results(self, name, i, p, E, psi, E_approx, psi_residual_magnitude):
        H = self.hamiltonians[name]
        H.E_valid_true[i] = E
        H.E_valid_approx[i] = E_approx
        H.p_valid[i] = p
        H.X_valid[:, i] = psi
        H.X_valid_residual_magnitude[i] = psi_residual_magnitude
        return self


OBSERVABLE_MAP = dict(
    Hamiltonian=Hamiltonian,
    Operator=Operator,
    RadiusOperator=RadiusOperator,
    TritonHalfLifeOperator=TritonHalfLifeOperator,
    MultiHamiltonianTrainer=MultiHamiltonianTrainer
)


def to_hdf5(obj, filename, mode="w", **kwargs):
    obj._to_hdf5(filename=filename, mode=mode, **kwargs)


def from_hdf5(filename, path="/", safe=True):
    import hickle
    with open(filename, "r") as f:
        state: dict = hickle.load(f, path=path, safe=safe)
    cls = OBSERVABLE_MAP[state["class_name"]]
    return cls.from_dict(state["attributes"])
