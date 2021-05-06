import numpy as np
from numpy import diag
from numpy.linalg import pinv
import numba as nb
from numba import jit


def pinv_eigenvalues(vals, i):
    val = vals[i]
    with np.errstate(divide="ignore"):
        vals_pinv = 1.0 / (vals - val)
    vals_pinv[i] = 0.0
    vals_pinv = diag(vals_pinv)
    return vals_pinv


def generalized_inverse(vals, vecs, i):
    vals_pinv = pinv_eigenvalues(vals, i)
    return vecs @ vals_pinv @ vecs.conj().T


def grad_eigvalsh_standard(vecs, dA):
    d_vals = np.zeros((*dA.shape[:-2], vecs.shape[-1]), dtype="float64")
    for i in range(vecs.shape[-1]):
        vec_i = vecs[:, i]
        d_vals[..., i] = (vec_i.conj().T @ dA @ vec_i).real
    return d_vals


def grad_eigh_standard(vals, vecs, A, dA):
    # d_vals = np.diagonal(vecs.conj().T @ dA @ vecs, axis1=-2, axis2=-1)
    grad_dims = dA.shape[:-2]
    d_vals = np.zeros((*grad_dims, *vals.shape), dtype="complex128")
    d_vecs = np.zeros((*grad_dims, *vecs.shape), dtype="complex128")
    for i, val in enumerate(vals):
        # This works and is simple but has numerical instability sometimes. Do it better below.
        # Id = np.eye(*A.shape)
        # d_vecs[..., i] = -pinv(A - val * Id, hermitian=True, rcond=1e-10) @ dA @ vecs[:, i]

        vec_i = vecs[:, i]
        d_vals[..., i] = vec_i.conj().T @ dA @ vec_i
        gen_inv = generalized_inverse(vals, vecs, i)
        d_vecs[..., i] = -gen_inv @ dA @ vec_i
    return d_vals, d_vecs


def grad_eigh_general(vals, vecs, dA, dB):
    grad_dims = dA.shape[:-2]
    d_vals = np.zeros((*grad_dims, *vals.shape), dtype="complex128")
    d_vecs = np.zeros((*grad_dims, *vecs.shape), dtype="complex128")
    for i, val in enumerate(vals):
        vec_i = vecs[:, i]
        vec_i_dagger = vec_i.conj().T
        # Gradient of eigenvalue i
        d_vals[..., i] = vec_i_dagger @ (dA - val * dB) @ vec_i

        # Gradient of eigenvector i
        gen_inv = generalized_inverse(vals, vecs, i)
        d_vecs[..., i] = (
                -gen_inv @ (dA - val * dB) @ vec_i
                - 0.5 * (vec_i_dagger @ dB @ vec_i) * vec_i
        )
    return d_vals, d_vecs


def grad_eigh_ritz(vals, vecs, dH, dN):
    # d_vals = np.diagonal(vecs.conj().T @ dH @ vecs, axis1=-2, axis2=-1)
    grad_dims = dH.shape[:-2]
    d_vals = np.zeros((*grad_dims, *vals.shape), dtype=dH.dtype)
    d_vecs = np.zeros((*grad_dims, *vecs.shape), dtype=dH.dtype)
    for i, val in enumerate(vals):
        vec_i = vecs[:, i]
        d_vals[..., i] = vec_i.conj().T @ dH @ vec_i
        gen_inv = generalized_inverse(vals, vecs, i)
        d_vecs[..., i] = (
                -gen_inv @ dH @ vec_i - 0.5 * (vec_i.conj().T @ dN @ vec_i)[:, None] * vec_i
        )
    return d_vals, d_vecs


@jit(
    # nb.complex128[:, :](
    #     nb.complex128[:, :],
    #     nb.complex128[:, :, :],
    #     nb.float64,
    #     nb.complex128[:],
    #     nb.float64[:],
    #     nb.float64,
    #     nb.float64,
    #     nb.int32,
    #     nb.int32,
    #     nb.boolean,
    # ),
    nopython=True,
)
def _grad_eigenvectors_hermitian_iterative_loop(
        H, dH, val, constant_term, precision, max_precise_iters, max_iters
):
    n_iters = 0
    n_precise_iters = 0
    d_vec = np.zeros(dH.shape[:-1], dtype=H.dtype)
    d_vec_new = np.zeros(dH.shape[:-1], dtype=H.dtype)
    while (n_precise_iters < max_precise_iters) and (n_iters < max_iters):
        n_iters += 1
        d_vec_new[...] = (constant_term + (d_vec.conj() @ H).conj()) / val

        max_residual = np.max(np.abs(d_vec_new - d_vec))
        if max_residual < precision:
            n_precise_iters += 1
        else:
            n_precise_iters = 0  # Reset it

        d_vec[...] = d_vec_new
    return d_vec, n_iters


def grad_eigenvectors_hermitian_iterative(
        H,
        dH,
        val,
        vec,
        d_val,
        offset=0,
        precision=1e-13,
        max_precise_iters=10,
        max_iters=10000,
        verbose=False,
):
    H = H - offset * np.eye(H.shape[0])
    val = val - offset
    constant_term = dH @ vec - d_val[:, None] * vec
    d_vec, n_iters = _grad_eigenvectors_hermitian_iterative_loop(
        H=H,
        dH=dH,
        val=val,
        constant_term=constant_term,
        precision=precision,
        max_precise_iters=max_precise_iters,
        max_iters=max_iters,
    )
    if verbose:
        print("It took", n_iters, "iterations to converge")
    return d_vec
