import numpy as np
from scipy import stats
import emcee
from sklearn.utils import check_random_state
from numpy.polynomial.legendre import leggauss
from numba import njit, prange


def invchi2(df, scale, **kwargs):
    a = df / 2
    b = df * scale ** 2 / 2
    return stats.invgamma(a, scale=b, **kwargs)


def leggauss_shifted(deg, a=-1, b=1):
    """Obtain the Gaussian quadrature points and weights when the limits of integration are [a, b]

    Parameters
    ----------
    deg : int
        The degree of the quadrature
    a : float
        The lower limit of integration. Defaults to -1, the standard value.
    b : float
        The upper limit of integration. Defaults to +1, the standard value.

    Returns
    -------
    x : The integration locations
    w : The weights
    """
    x, w = leggauss(deg)
    w *= (b - a) / 2.0
    x = ((b - a) * x + (b + a)) / 2.0
    return x, w


def compute_coefficients_pure_python(y, Q, ref, orders=None, ignore_orders=None):
    if y.ndim != 2:
        raise ValueError("y must be 2d")
    if orders is None:
        orders = np.arange(y.shape[-1])
    if len(orders) != y.shape[-1]:
        raise ValueError("y and orders must have the same length")

    ref = np.atleast_2d(np.asarray(ref)).T
    Q = np.atleast_2d(np.asarray(Q)).T
    orders = np.atleast_1d(orders)

    # Make coefficients
    coeffs = np.diff(y, axis=-1)  # Find differences
    coeffs = np.insert(coeffs, 0, y[..., 0], axis=-1)  # But keep leading term
    coeffs = coeffs / (ref * Q ** orders)  # Scale each order appropriately
    if ignore_orders is not None:
        coeffs = coeffs[:, ~np.isin(orders, ignore_orders)]
    return coeffs


@njit(fastmath=True)
def n_isin(a, n):
    for i in range(len(a)):
        if a[i] == n:
            return True
    return False


@njit
def compute_coefficients(y, Q, ref, orders=None, ignore_orders=None):
    if y.ndim != 2:
        raise ValueError("y must be 2d")
    if orders is None:
        orders = np.arange(y.shape[-1])
    if len(orders) != y.shape[-1]:
        raise ValueError("y and orders must have the same length")

    ref = np.atleast_2d(np.asarray(ref)).T
    Q = np.atleast_2d(np.asarray(Q)).T
    orders = np.atleast_1d(orders)

    # Make coefficients
    coeffs = np.zeros(y.shape)
    orders_used = []
    idx_used = []
    for n, y_n in enumerate(y.T):
        order = orders[n]
        if ignore_orders is not None and n_isin(ignore_orders, order):
            continue
        if n == 0:
            coeffs[:, n] = y_n
        else:
            coeffs[:, n] = y_n - y[:, n - 1]
        orders_used.append(order)
        idx_used.append(n)
    idx_used = np.array(idx_used)
    coeffs = coeffs / (ref * Q ** orders)  # Scale each order appropriately
    if ignore_orders is not None:
        coeffs = coeffs[:, idx_used]
    return coeffs


@njit
def compute_degrees_of_freedom(coeffs, df_0):
    return df_0 + coeffs.size


@njit
def compute_scale(coeffs, df_0, scale_0):
    df = compute_degrees_of_freedom(coeffs, df_0)
    return np.sqrt((df_0 * scale_0 ** 2 + np.sum(coeffs ** 2)) / df)


# def compute_unnormalized_Q_posterior(
#     y, Q, ref, orders, df_0, scale_0, Q_prior, ignore_orders=None
# ):
#     coeffs = compute_coefficients(
#         y=y, Q=Q, ref=ref, orders=orders, ignore_orders=ignore_orders
#     )
#     df = compute_degrees_of_freedom(coeffs, df_0)
#     scale = compute_scale(coeffs, df_0, scale_0)
#     n_obs = y.shape[0]
#     if ignore_orders is None:
#         ignore_orders = []
#     Q_prod = np.product(
#         [np.abs(Q ** (n_obs * n)) for n in orders if n not in ignore_orders]
#     )
#     Q_post_raw = Q_prior(Q) / (scale ** df * Q_prod)
#     return Q_post_raw


def compute_cbar2_given_Q_logpdf(
    y, cbar, Q, ref, orders, df_0, scale_0, ignore_orders=None
):
    coeffs = compute_coefficients(
        y=y, Q=Q, ref=ref, orders=orders, ignore_orders=ignore_orders
    )
    df = compute_degrees_of_freedom(coeffs, df_0)
    scale = compute_scale(coeffs, df_0, scale_0)
    dist = invchi2(df=df, scale=scale)
    return dist.logpdf(cbar ** 2)


@njit
def compute_unnormalized_Q_logpdf(
    y, Q, ref, orders, df_0, scale_0, Q_log_prior, ignore_orders=None
):
    coeffs = compute_coefficients(
        y=y, Q=Q, ref=ref, orders=orders, ignore_orders=ignore_orders
    )
    df = compute_degrees_of_freedom(coeffs, df_0)
    scale = compute_scale(coeffs, df_0, scale_0)
    n_obs = y.shape[0]
    # if ignore_orders is None:
    #     ignore_orders = []
    # log_Q_prod = np.sum(
    #     [(n_obs * n) * np.log(np.abs(Q)) for n in orders if n not in ignore_orders]
    # )
    log_Q_prod = 0
    for n in orders:
        if ignore_orders is None or not n_isin(ignore_orders, n):
            log_Q_prod += (n_obs * n) * np.log(np.abs(Q))
    return Q_log_prior - df * np.log(scale) - log_Q_prod


@njit(parallel=True)
def compute_parallel_unnormalized_Q_logpdf(
    y, Q_array, ref, orders, df_0, scale_0, Q_log_prior_array, ignore_orders=None
):
    logpdfs = np.zeros(len(Q_array))
    for i in prange(len(Q_array)):
        Q_i = Q_array[i]
        Q_log_prior_i = Q_log_prior_array[i]
        logpdfs[i] = compute_unnormalized_Q_logpdf(
            y,
            Q_i,
            ref,
            orders,
            df_0,
            scale_0,
            Q_log_prior=Q_log_prior_i,
            ignore_orders=ignore_orders,
        )
    return logpdfs


class TruncationDistribution:
    def __init__(
        self,
        df_0,
        scale_0,
        a_0,
        b_0,
        y_lower,
        orders_lower,
        ignore_orders=None,
        y_ref=1,
        update_prior=True,
        deg_quadrature=80,
    ):
        self.df_0 = df_0
        self.scale_0 = scale_0
        self.a_0 = a_0
        self.b_0 = b_0
        self.y_lower = y_lower
        self.orders_lower = orders_lower
        if ignore_orders is None:
            ignore_orders = []
        self.ignore_orders = np.array(ignore_orders)
        self.y_ref = y_ref
        self.update_prior = update_prior

        self.cbar_sq_prior_dist = invchi2(df=df_0, scale=scale_0)
        self.Q_prior_dist = stats.beta(a=a_0, b=b_0)
        self.deg_quadrature = deg_quadrature
        self.x_quadrature, self.w_quadrature = leggauss_shifted(
            deg_quadrature, a=0, b=1
        )

    def logpdf(self, cbar, Q, y=None, order=None):
        if self.update_prior:
            if y is not None:
                y = np.atleast_1d(y)
                if y.ndim == 1:
                    y = y[:, None]
            if self.y_lower is not None and y is not None:
                y = np.hstack([self.y_lower, y])
                orders = np.append(self.orders_lower, order)
            elif self.y_lower is not None:
                y = self.y_lower
                orders = self.orders_lower
            elif y is not None:
                orders = np.atleast_1d(order)
            else:
                raise ValueError(
                    "Some data (y or y_lower) must be given if update_prior is True"
                )

            cbar_sq_given_Q_logpdf = compute_cbar2_given_Q_logpdf(
                y,
                cbar,
                Q,
                ref=self.y_ref,
                orders=orders,
                df_0=self.df_0,
                scale_0=self.scale_0,
                ignore_orders=self.ignore_orders,
            )

            Q_logpdf_unnorm = compute_unnormalized_Q_logpdf(
                y,
                Q,
                ref=self.y_ref,
                orders=orders,
                df_0=self.df_0,
                scale_0=self.scale_0,
                Q_log_prior=self.Q_prior_dist.logpdf(Q),
                ignore_orders=self.ignore_orders,
            )

            Q_pdf_unnorm_vals = np.exp(
                compute_parallel_unnormalized_Q_logpdf(
                    y,
                    Q_array=self.x_quadrature,
                    ref=self.y_ref,
                    orders=orders,
                    df_0=self.df_0,
                    scale_0=self.scale_0,
                    Q_log_prior_array=self.Q_prior_dist.logpdf(self.x_quadrature),
                    ignore_orders=self.ignore_orders,
                )
            )
            Q_norm = np.sum(self.w_quadrature * Q_pdf_unnorm_vals)
            Q_logpdf = Q_logpdf_unnorm - np.log(Q_norm)
            return cbar_sq_given_Q_logpdf + Q_logpdf
        else:
            cbar_sq_logprior = self.cbar_sq_prior_dist.logpdf(cbar ** 2)
            Q_logprior = self.Q_prior_dist.logpdf(Q)
            return cbar_sq_logprior + Q_logprior

    def logprior(self, cbar, Q):
        return self.cbar_sq_prior_dist.logpdf(cbar ** 2) + self.Q_prior_dist.logpdf(Q)

    def cbar_sq_logprior(self, cbar):
        return self.cbar_sq_prior_dist.logpdf(cbar ** 2)

    def Q_logprior(self, Q):
        return self.Q_prior_dist.logpdf(Q)

    def cbar_sq_prior(self, cbar):
        return self.cbar_sq_prior_dist.pdf(cbar ** 2)

    def Q_prior(self, Q):
        return self.Q_prior_dist.pdf(Q)


class ObservableDistribution:
    def __init__(
        self,
        data,
        observables,
        order,
        cov_expt,
        mean_posterior_lecs=None,
        cov_posterior_lecs=None,
        lecs_prior_std=1,
        trunc_dist=None,
        trunc_ignore_observables=None,
        cbar=None,
        Q=None,
        return_predictions=True,
        output_file=None,
        observable_names=None,
        parameter_names=None,
        posterior_lecs_indices=None,
        compute_gradients=False,
    ):
        self.data = data
        self.observables = observables
        if observable_names is None:
            observable_names = [str(obs) for obs in observables]
        if len(observable_names) != len(observables):
            raise ValueError(
                "The length of observables must match the length of observable_names."
            )
        self.observable_names = observable_names
        self.n_obs = len(observables)
        self.n_lecs = observables[0].n_lecs
        self.cov_expt = cov_expt
        self.cov_posterior_lecs = cov_posterior_lecs
        self.posterior_lecs_indices = posterior_lecs_indices
        self.prior_lecs_indices = None

        self.n_posterior_lecs = None
        self.compute_gradients = False
        self.lecs_posterior = None
        self.lecs_prior = stats.norm(scale=lecs_prior_std)
        if cov_posterior_lecs is not None:
            self.n_posterior_lecs = len(cov_posterior_lecs)
            self.compute_gradients = compute_gradients
            if posterior_lecs_indices is not None:
                self.prior_lecs_indices = [
                    n for n in range(self.n_lecs) if n not in posterior_lecs_indices
                ]
                self.lecs_posterior = stats.multivariate_normal(
                    mean=mean_posterior_lecs, cov=cov_posterior_lecs
                )
        else:
            if compute_gradients:
                raise ValueError(
                    "cov_posterior_lecs is required if compute_gradients is true"
                )
        self.order = order

        self.yref = np.abs(data)
        self.cbar = cbar
        self.Q = Q
        cbar_idx = Q_idx = None
        if cbar is None:
            cbar_idx = 0
            if Q is None:
                Q_idx = 1
        elif Q is None:
            Q_idx = 0

        self.cbar_idx = cbar_idx
        self.Q_idx = Q_idx
        self.hyper_indices = [idx for idx in [cbar_idx, Q_idx] if idx is not None]
        self.sampler = None
        self.return_predictions = return_predictions
        self.output_file = output_file
        self.parameter_names = parameter_names if parameter_names is not None else []
        self.parameter_names_with_hyperparameters = []
        if parameter_names is not None:
            parameter_names_with_hyperparameters = []
            if cbar_idx is not None:
                parameter_names_with_hyperparameters.append("cbar")
            if Q_idx is not None:
                parameter_names_with_hyperparameters.append("Q")
            self.parameter_names_with_hyperparameters = (
                parameter_names_with_hyperparameters + parameter_names
            )

        if trunc_dist is None:
            trunc_dist = TruncationDistribution(
                df_0=1,
                scale_0=1,
                a_0=1,
                b_0=1,
                y_lower=None,
                orders_lower=None,
                ignore_orders=None,
                y_ref=self.yref,
                update_prior=False,
            )
        if len(self.hyper_indices) == 0:
            trunc_dist = None
        self.trunc_dist = trunc_dist

        trunc_dist_obs_mask = np.ones(len(observable_names), dtype=bool)
        if trunc_ignore_observables is not None:
            not_in_obs_names = ~np.isin(trunc_ignore_observables, observable_names)
            trunc_dist_obs_mask = ~np.isin(observable_names, trunc_ignore_observables)
            if np.any(not_in_obs_names):
                print(
                    f"The ignored observables are not all in the list of observable names.\n"
                    f"trunc_ignore_observables = {trunc_ignore_observables}\n"
                    f"Bad indices: {not_in_obs_names}\n"
                    f"observable_names = {observable_names}"
                )
                # raise ValueError(
                #     f"The ignored observables are not all in the list of observable names.\n"
                #     f"trunc_ignore_observables = {trunc_ignore_observables}\n"
                #     f"Bad indices: {not_in_obs_names}\n"
                #     f"observable_names = {observable_names}"
                # )
        self.trunc_ignore_observables = trunc_ignore_observables
        self.trunc_dist_obs_mask = trunc_dist_obs_mask

    def __call__(self, p):
        cbar = self.cbar
        Q = self.Q
        if cbar is None:
            cbar = p[self.cbar_idx]
        if Q is None:
            Q = p[self.Q_idx]
        lecs = np.delete(p, self.hyper_indices)
        return self.logpdf(lecs, cbar, Q, self.return_predictions)

    def logpdf(self, lecs, cbar, Q, return_predictions=False):
        if cbar < 0 or Q <= 0 or Q >= 1:
            preds = self.compute_observables(lecs, return_gradient=False)
            out = -np.inf, *list(preds)
            return out
        log_like, pred = self.loglike(lecs, cbar=cbar, Q=Q, return_predictions=True)
        log_pdf = log_like + self.logprior(lecs, cbar=cbar, Q=Q, y=pred)
        if return_predictions:
            out = log_pdf, *list(pred)
            return out
        return log_pdf

    def compute_observables(self, lecs, return_gradient=False):
        preds = np.zeros(self.n_obs)
        if return_gradient:
            grads = np.zeros((self.n_posterior_lecs, self.n_obs))
            for i, obs in enumerate(self.observables):
                preds[i], grads[..., i] = obs(lecs, return_gradient=return_gradient)
            return preds, grads
        else:
            for i, obs in enumerate(self.observables):
                preds[i] = obs(lecs, return_gradient=return_gradient)
            return preds

    def loglike(self, lecs, cbar, Q, return_predictions=False):
        cov_trunc = self.compute_truncation_cov(cbar=cbar, Q=Q)
        cov_like = self.cov_expt + cov_trunc
        if self.compute_gradients:
            preds, grads = self.compute_observables(lecs, return_gradient=True)
            cov_grad = grads.T @ self.cov_posterior_lecs @ grads
            cov_like = cov_like + cov_grad
        else:
            preds = self.compute_observables(lecs, return_gradient=False)

        try:
            dist = stats.multivariate_normal(
                mean=self.data, cov=cov_like, allow_singular=False
            )
            log_like = dist.logpdf(preds)
        except np.linalg.LinAlgError:
            print(
                f"Warning: The covariance matrix is singular. lecs: {lecs}, cbar: {cbar}, Q: {Q}"
            )
            log_like = -np.inf
        if return_predictions:
            return log_like, preds
        return log_like

    def logprior(self, lecs, cbar, Q, y):
        if cbar < 0:
            return -np.inf
        if self.posterior_lecs_indices is not None:
            pr_lecs = (
                self.lecs_posterior.logpdf(lecs[self.posterior_lecs_indices])
                + self.lecs_prior.logpdf(lecs[self.prior_lecs_indices]).sum()
            )
        else:
            pr_lecs = self.lecs_prior.logpdf(lecs).sum()
        if self.trunc_dist is None:
            return pr_lecs
        y = y[self.trunc_dist_obs_mask]
        return (
            pr_lecs
            + self.trunc_dist.logpdf(cbar=cbar, Q=Q, y=y, order=self.order).sum()
        )

    def compute_truncation_cov(self, cbar, Q):
        k = self.order
        y_ref = self.yref
        if Q >= 1:
            Q_std = 1e10
        else:
            Q_std = Q ** (k + 1) / np.sqrt(1 - Q ** 2)
        return np.diag(y_ref * cbar * Q_std) ** 2

    def sample(
        self,
        nwalkers,
        n_burn_in=200,
        n_samples=1500,
        seed=0,
        pool=None,
        backend=None,
        progress=True,
    ):
        r"""A wrapper for MCMC sampling that allows this object to store the predictions made along the way

        After calling, checkout `self.pred_samples`. The emcee sampler is stored in `self.sampler`.
        """
        ndim = self.n_lecs + len(self.hyper_indices)
        blobs_dtype = None
        if self.return_predictions:
            blobs_dtype = [(name, float) for name in self.observable_names]

        if self.output_file is not None:
            backend = emcee.backends.HDFBackend(self.output_file)

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
            nwalkers=nwalkers,
            ndim=ndim,
            log_prob_fn=self,
            pool=pool,
            blobs_dtype=blobs_dtype,
            backend=backend,
        )

        rng = check_random_state(seed)
        p0 = rng.rand(nwalkers, ndim)
        sampler.random_state = seed  # Try setting the state. Maybe it will work?
        if progress:
            print("Running burn-in", flush=True)
        state = sampler.run_mcmc(p0, n_burn_in, store=False, progress=progress)

        sampler.reset()
        # run a lot of samples
        if progress:
            print("Running sampler", flush=True)
        sampler.run_mcmc(state, n_samples, progress=progress)
        self.sampler = sampler
        if self.output_file is not None and progress:
            print(f"Output saved to {self.output_file}")
        return self

    def get_chain(self, **kwargs):
        return self.sampler.get_chain(**kwargs)

    def get_blobs(self, **kwargs):
        return self.sampler.get_blobs(**kwargs)


def adjust(x, lo, hi):
    return x * (hi - lo) + lo
