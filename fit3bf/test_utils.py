import numpy as np
from sklearn.utils import check_random_state
import tensorflow as tf
from numpy import diag
from numpy.linalg import pinv
import numba as nb
from numba import jit


def create_random_symmetric_matrix(n, random_state=None, use_tf=False):
    rng = check_random_state(random_state)
    mat = rng.rand(n, n)
    mat = mat + mat.T + 1e-5 * np.eye(n)
    if use_tf:
        mat = tf.Variable(
            tf.constant(mat, dtype=tf.complex128), name=f"sym_mat_{n}_{random_state}"
        )
    return mat


def create_random_psd_matrix(n, random_state=None, use_tf=False, is_hermitian=False):
    rng = check_random_state(random_state)
    mat = rng.rand(n, n)
    if is_hermitian:
        mat = mat + 1j * rng.rand(n, n)
    mat = (mat + mat.conj().T) / 2.0 + n * np.eye(n)
    if use_tf:
        mat = tf.Variable(
            tf.constant(mat, dtype=tf.complex128), name=f"psd_mat_{n}_{random_state}"
        )
    return mat


def create_random_hermitian_matrix(n, random_state=None, use_tf=False):
    rng = check_random_state(random_state)
    mat = rng.rand(n, n) + 1j * rng.rand(n, n)
    mat = mat + mat.conj().T + 1e-5 * np.eye(n)
    if use_tf:
        mat = tf.Variable(
            tf.constant(mat, dtype=tf.complex128), name=f"her_mat_{n}_{random_state}"
        )
    return mat


def create_random_polynomial_matrix(
    n,
    x,
    coefficients,
    random_state=None,
    is_hermitian=False,
    use_tf=False,
    take_grad=False,
):
    if is_hermitian:
        dtype = "complex128"
    else:
        dtype = "complex128"
    mat = np.zeros((n, n), dtype=dtype)
    if random_state is None:
        random_state = 0
    if use_tf:
        mat = tf.convert_to_tensor(mat)
    for i, c in enumerate(coefficients):
        if take_grad:
            pre_factor = c * i * x ** (i - 1)
        else:
            pre_factor = c * x ** i
        if use_tf:
            pre_factor = tf.cast(pre_factor, dtype=tf.complex128)
        if is_hermitian:
            next_term = pre_factor * create_random_hermitian_matrix(
                n, random_state=random_state + i, use_tf=use_tf
            )
        else:
            next_term = pre_factor * create_random_symmetric_matrix(
                n, random_state=random_state + i, use_tf=use_tf
            )
        mat += next_term

    #     if take_grad:
    #         pre_factor = c * i * x ** (i - 1)
    #     else:
    #         pre_factor = c * x ** i
    # if is_hermitian:
    #     mat = x * create_random_hermitian_matrix(
    #         n, random_state=random_state, use_tf=use_tf
    #     )
    # else:
    #     mat = x * create_random_symmetric_matrix(
    #         n, random_state=random_state, use_tf=use_tf
    #     )
    if take_grad:
        mat = np.array([mat])
    return mat


def create_random_polynomial_psd_matrix(
    n,
    x,
    coefficients,
    random_state=None,
    is_hermitian=False,
    use_tf=False,
    take_grad=False,
):
    if is_hermitian:
        dtype = "complex128"
    else:
        dtype = "complex128"
    mat = np.zeros((n, n), dtype=dtype)
    if random_state is None:
        random_state = 0
    if use_tf:
        mat = tf.convert_to_tensor(mat)
    for i, c in enumerate(coefficients):
        if take_grad:
            pre_factor = c * i * x ** (i - 1)
        else:
            pre_factor = c * x ** i
        if use_tf:
            pre_factor = tf.cast(pre_factor, dtype=tf.complex128)
        next_term = pre_factor * create_random_psd_matrix(
            n, random_state=random_state + i, use_tf=use_tf, is_hermitian=is_hermitian
        )
        mat += next_term
    if take_grad:
        mat = np.array([mat])
    return mat

# def grad_eigh_standard_tensorflow(A, x):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch([x, A])
#         e_val, e_vec = tf.linalg.eigh(A)
#
#     experimental_use_pfor = True
#     eval_grad = tape.jacobian(e_val, x, experimental_use_pfor=experimental_use_pfor)
#     evec_grad = tape.jacobian(e_vec, x, experimental_use_pfor=experimental_use_pfor)
#     return eval_grad, evec_grad


def grad_eigh_tensorflow(
    n, x, coefficients, random_state=None, is_hermitian=False, is_generalized=False
):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x])
        A = create_random_polynomial_matrix(
            n,
            x,
            coefficients,
            random_state=random_state,
            is_hermitian=is_hermitian,
            use_tf=True,
            take_grad=False,
        )
        if is_generalized:
            # if is_hermitian:
            #     B = create_random_hermitian_matrix(n, random_state, use_tf=True)
            # else:
            #     B = create_random_symmetric_matrix(n, random_state, use_tf=True)
            # B = create_random_psd_matrix(n, random_state, use_tf=False)
            # B = np.eye(n)
            # B = tf.Variable(B, name='B', dtype=tf.complex128)
            coefficients_b = coefficients
            B = create_random_polynomial_psd_matrix(
                n=n,
                x=x,
                coefficients=coefficients_b,
                random_state=23 * (np.abs(random_state) + 4),
                use_tf=True,
                take_grad=False,
                is_hermitian=is_hermitian,
            )
            # sqrt_N = tf.linalg.sqrtm(B, name="sqrt_N")
            # inv_sqrt_N = tf.linalg.inv(sqrt_N)
            # print(inv_sqrt_N)
            # inv_sqrt_N_conj = inv_sqrt_N
            # # inv_sqrt_N_conj = tf.linalg.adjoint(inv_sqrt_N)
            # vals, vecs_trans = tf.linalg.eigh(inv_sqrt_N @ A @ inv_sqrt_N_conj)
            # vecs = inv_sqrt_N_conj @ vecs_trans

            sqrt_N = tf.linalg.cholesky(B, name="chol_N")
            inv_sqrt_N = tf.linalg.inv(sqrt_N)
            print("inv sqrt N")
            print(inv_sqrt_N)
            inv_sqrt_N_conj = tf.linalg.adjoint(inv_sqrt_N)
            vals, vecs_trans = tf.linalg.eigh(inv_sqrt_N @ A @ inv_sqrt_N_conj)
            vecs = inv_sqrt_N_conj @ vecs_trans
        else:
            vals, vecs = tf.linalg.eigh(A)

        vals = tf.cast(vals, dtype=tf.complex128)
        vecs = tf.cast(vecs, dtype=tf.complex128)

        vals_times_i = 1j * vals
        vecs_times_i = 1j * vecs

    # TODO: Figure out wtf is wrong with TensorFlow here. It casts everything to floats
    #       when x is real, and then only keeps the real parts of the output gradients.
    #       When x is complex, I get weird results that don't match with my analytic code
    #       (This could be due to how TF handles gradients wrt complex numbers.)
    #       I get around this by computing the gradient of 1j times the eigensystem as well,
    #       and then adding their outputs together at the end. Since the matrix is Hermitian,
    #       this should not matter for the eigenvalues, but will matter for the eigenvectors
    #       because they can have complex entries.
    #       This only works when `experimental_use_pfor = True`, otherwise an error is thrown.

    experimental_use_pfor = True

    d_vals = tf.convert_to_tensor(
        tape.jacobian(vals, [x], experimental_use_pfor=experimental_use_pfor),
        dtype=tf.complex128,
    )
    d_vecs = tf.convert_to_tensor(
        tape.jacobian(vecs, [x], experimental_use_pfor=experimental_use_pfor),
        dtype=tf.complex128,
    )

    d_vals_times_i = tf.convert_to_tensor(
        tape.jacobian(vals_times_i, [x], experimental_use_pfor=experimental_use_pfor),
        dtype=tf.complex128,
    )
    d_vecs_times_i = tf.convert_to_tensor(
        tape.jacobian(vecs_times_i, [x], experimental_use_pfor=experimental_use_pfor),
        dtype=tf.complex128,
    )
    d_vals = d_vals - 1.0j * d_vals_times_i
    d_vecs = d_vecs - 1.0j * d_vecs_times_i
    return vals, vecs, d_vals, d_vecs


# def grad_eigh_general_tensorflow(
#     n, x, coefficients, random_state=None, is_hermitian=False
# ):
#     # with tf.GradientTape(persistent=True) as g:
#     #     with tf.GradientTape(persistent=True) as g2:
#     #         g.watch([x])
#     #         sqrt_N = tf.linalg.sqrtm(B, name="sqrt_N")
#     #         inv_sqrt_N = tf.linalg.inv(sqrt_N)
#     #         e_val, e_vec_trans = tf.linalg.eigh(inv_sqrt_N @ A @ inv_sqrt_N)
#     #         e_vec = inv_sqrt_N @ e_vec_trans
#
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch([x])
#         A = create_random_polynomial_matrix(
#             n,
#             x,
#             coefficients,
#             random_state=random_state,
#             is_hermitian=is_hermitian,
#             use_tf=True,
#             take_grad=False,
#         )
#         # if is_hermitian:
#         #     B = create_random_hermitian_matrix(n, random_state, use_tf=True)
#         # else:
#         #     B = create_random_symmetric_matrix(n, random_state, use_tf=True)
#         B = create_random_psd_matrix(n, random_state, use_tf=False)
#         sqrt_N = tf.linalg.sqrtm(B, name="sqrt_N")
#         inv_sqrt_N = tf.linalg.inv(sqrt_N)
#         e_val, e_vec_trans = tf.linalg.eigh(inv_sqrt_N @ A @ inv_sqrt_N)
#         e_vec = inv_sqrt_N @ e_vec_trans
#
#     print([var.name for var in tape.watched_variables()])
#     experimental_use_pfor = False
#     eval_grad = tape.jacobian(e_val, x, experimental_use_pfor=experimental_use_pfor)
#     evec_grad = tape.jacobian(e_vec, x, experimental_use_pfor=experimental_use_pfor)
#     return eval_grad, evec_grad
