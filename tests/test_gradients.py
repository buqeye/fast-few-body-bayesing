import pytest
from fit3bf.test_utils import *
from fit3bf.gradients import *
import tensorflow as tf
import numpy as np
from scipy.linalg import eigh

# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(False)

c = [1, 3, 2, 9]
# theta = np.linspace(0, 10.0, 10)
theta = 2


@pytest.mark.parametrize(
    "size, random_state, x, coefficients, is_hermitian",
    [(5, 70, theta, c, True), (10, 5, theta, c, True), (13, 15, 52, c, True)],
)
def test_eigh(size, random_state, x, coefficients, is_hermitian):
    A = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=False,
        is_hermitian=is_hermitian,
    )
    vals_sp, vecs_sp = eigh(A)
    vals, vecs = np.linalg.eigh(A)

    relative_phase = np.angle(vecs[0, :] / vecs_sp[0, :])
    vecs_sp_normalized = vecs_sp / np.exp(1j * relative_phase)

    np.testing.assert_allclose(vals_sp, vals)
    np.testing.assert_allclose(vecs_sp_normalized, vecs)


@pytest.mark.parametrize(
    "size, random_state, x, coefficients, is_hermitian",
    [
        (5, 70, theta, c, True),
        (25, 70, theta, c, False),
        (10, 5, theta, c, True),
        (13, 15, 52, c, True),
    ],
)
def test_gradients_standard(size, random_state, x, coefficients, is_hermitian):
    A = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=False,
        is_hermitian=is_hermitian,
    )
    dA = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=True,
        is_hermitian=is_hermitian,
    )
    np.testing.assert_allclose(A, A.conj().T)
    np.testing.assert_allclose(dA, np.transpose(dA.conj(), axes=(0, 2, 1)))
    vals, vecs = eigh(A)
    # vals, vecs = np.linalg.eigh(A)
    d_vals, d_vecs = grad_eigh_standard(vals, vecs, A=A, dA=dA)

    dtype_tf = tf.float64
    # if is_hermitian:
    #     dtype_tf = tf.complex128

    # g = tf.Graph()
    # with g.as_default():
    #     x_tf = tf.Variable(initial_value=x, name="x", dtype=dtype_tf)
    #     vals_tf, vecs_tf, d_vals_tf, d_vecs_tf = grad_eigh_standard_tensorflow(
    #         n=size,
    #         x=x_tf,
    #         coefficients=coefficients,
    #         random_state=random_state,
    #         is_hermitian=is_hermitian,
    #     )
    #
    #     # Eigenvectors are only defined up to a phase.
    #     # Accounting for the phase difference in the vectors *should* fix any
    #     # discrepancies in their gradients
    #     # relative_phase = np.angle(vecs[0, :] / vecs_tf.numpy()[0, :])
    #     relative_phase = 0.0
    #     with v1.Session(graph=g) as sess:
    #         sess.run(v1.global_variables_initializer())
    #         vals_tf_eval, vecs_tf_eval, d_vals_tf_eval, d_vecs_tf_eval = sess.run(
    #             [vals_tf, vecs_tf, d_vals_tf, d_vecs_tf], feed_dict={x_tf: x}
    #         )
    #         print(vals_tf_eval, vecs_tf_eval, d_vals_tf_eval, d_vecs_tf_eval)

    x_tf = tf.Variable(initial_value=x, name="x", dtype=dtype_tf)
    vals_tf, vecs_tf, d_vals_tf, d_vecs_tf = grad_eigh_tensorflow(
        n=size,
        x=x_tf,
        coefficients=coefficients,
        random_state=random_state,
        is_hermitian=is_hermitian,
    )

    d_vals_tf_eval = d_vals_tf.numpy()
    vecs_tf_eval = vecs_tf.numpy()
    d_vecs_tf_eval = d_vecs_tf.numpy()
    # d_vecs_tf_eval = tf.keras.backend.eval(d_vecs_tf)

    # Eigenvectors are only defined up to a phase.
    # Accounting for the phase difference in the vectors *should* fix any
    # discrepancies in their gradients
    relative_phase = np.angle(vecs[0, :] / vecs_tf_eval[0, :])
    d_vecs_tf_normalized = d_vecs_tf_eval / np.exp(1j * relative_phase)

    np.testing.assert_allclose(d_vals, d_vals_tf_eval)
    np.testing.assert_allclose(d_vecs, d_vecs_tf_normalized)


@pytest.mark.parametrize(
    "size, random_state, x, coefficients, is_hermitian",
    [(5, 0, theta, c, True), (10, 5, theta, c, False), (13, 15, 52, c, False)],
)
def test_gradients_generalized(size, random_state, x, coefficients, is_hermitian):
    A = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=False,
        is_hermitian=is_hermitian,
    )
    dA = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=True,
        is_hermitian=is_hermitian,
    )
    # if is_hermitian:
    #     B = create_random_hermitian_matrix(size, random_state, use_tf=False)
    # else:
    #     B = create_random_symmetric_matrix(size, random_state, use_tf=False)
    # B = create_random_psd_matrix(size, random_state, use_tf=False)
    # dB = np.zeros_like(dA)
    random_state_b = 23 * (np.abs(random_state) + 4)
    # coefficients_b = [1,2]
    coefficients_b = coefficients
    B = create_random_polynomial_psd_matrix(
        n=size,
        x=x,
        coefficients=coefficients_b,
        random_state=random_state_b,
        take_grad=False,
        is_hermitian=is_hermitian,
    )
    dB = create_random_polynomial_psd_matrix(
        n=size,
        x=x,
        coefficients=coefficients_b,
        random_state=random_state_b,
        take_grad=True,
        is_hermitian=is_hermitian,
    )
    # B = np.eye(size)
    print("")
    print(B)
    print(np.linalg.eigvalsh(B))
    vals, vecs = eigh(A, B)
    d_vals, d_vecs = grad_eigh_general(vals, vecs, dA=dA, dB=dB)

    dtype_tf = tf.float64
    # if is_hermitian:
    #     dtype_tf = tf.complex128

    # g = tf.Graph()
    # with g.as_default():
    #     x_tf = tf.Variable(initial_value=x, name="x", dtype=dtype_tf)
    #     vals_tf, vecs_tf, d_vals_tf, d_vecs_tf = grad_eigh_standard_tensorflow(
    #         n=size,
    #         x=x_tf,
    #         coefficients=coefficients,
    #         random_state=random_state,
    #         is_hermitian=is_hermitian,
    #     )
    #
    #     # Eigenvectors are only defined up to a phase.
    #     # Accounting for the phase difference in the vectors *should* fix any
    #     # discrepancies in their gradients
    #     # relative_phase = np.angle(vecs[0, :] / vecs_tf.numpy()[0, :])
    #     relative_phase = 0.0
    #     with v1.Session(graph=g) as sess:
    #         sess.run(v1.global_variables_initializer())
    #         vals_tf_eval, vecs_tf_eval, d_vals_tf_eval, d_vecs_tf_eval = sess.run(
    #             [vals_tf, vecs_tf, d_vals_tf, d_vecs_tf], feed_dict={x_tf: x}
    #         )
    #         print(vals_tf_eval, vecs_tf_eval, d_vals_tf_eval, d_vecs_tf_eval)

    x_tf = tf.Variable(initial_value=x, name="x", dtype=dtype_tf)
    vals_tf, vecs_tf, d_vals_tf, d_vecs_tf = grad_eigh_tensorflow(
        n=size,
        x=x_tf,
        coefficients=coefficients,
        random_state=random_state,
        is_hermitian=is_hermitian,
        is_generalized=True,
    )

    vals_tf_eval = vals_tf.numpy()
    d_vals_tf_eval = d_vals_tf.numpy()
    vecs_tf_eval = vecs_tf.numpy()
    d_vecs_tf_eval = d_vecs_tf.numpy()
    # d_vecs_tf_eval = tf.keras.backend.eval(d_vecs_tf)

    # Eigenvectors are only defined up to a phase.
    # Accounting for the phase difference in the vectors *should* fix any
    # discrepancies in their gradients
    # relative_phase = 0.0
    relative_phase = np.angle(vecs[0, :] / vecs_tf_eval[0, :])
    # relative_phase = np.angle(d_vecs[..., 0, :] / d_vecs_tf_eval[..., 0, :])
    vecs_tf_normalized = vecs_tf_eval / np.exp(1j * relative_phase)
    d_vecs_tf_normalized = d_vecs_tf_eval / np.exp(1j * relative_phase)

    np.testing.assert_allclose(vals, vals_tf_eval)
    np.testing.assert_allclose(vecs, vecs_tf_normalized)
    np.testing.assert_allclose(d_vals, d_vals_tf_eval)
    # np.testing.assert_allclose(np.abs(d_vecs), np.abs(d_vecs_tf_normalized))
    np.testing.assert_allclose(d_vecs, d_vecs_tf_normalized)


@pytest.mark.parametrize(
    "size, random_state, x, coefficients, is_hermitian",
    [
        (5, 70, theta, c, False),
        (25, 70, theta, c, True),
        (10, 5, theta, c, False),
        (13, 15, 52, c, True),
        (1000, 15, 52, c, True),
        (2000, 15, 52, c, True),
        # (3000, 15, 52, c, True),
    ],
)
def test_gradients_iterative(size, random_state, x, coefficients, is_hermitian):
    print("Making matrices")
    A = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=False,
        is_hermitian=is_hermitian,
    )
    dA = create_random_polynomial_matrix(
        n=size,
        x=x,
        coefficients=coefficients,
        random_state=random_state,
        take_grad=True,
        is_hermitian=is_hermitian,
    )
    print("Starting analytic method")
    vals, vecs = eigh(A)
    d_vals, d_vecs = grad_eigh_standard(vals, vecs, A=A, dA=dA)
    #
    # # idx = [np.argmax(np.abs(vals))]
    # # offset = 0
    idx = np.argmin(vals)
    offset = np.max(vals)
    # # offset = 0
    # print(vals, vals[idx])

    # d_vals = np.array([vals])
    # d_vecs = np.array([vecs])

    print("Starting iterative method")
    d_vec_iterative = grad_eigenvectors_hermitian_iterative(
        A, dA, val=vals[idx], vec=vecs[:, idx], d_val=d_vals[..., idx], offset=offset
    )
    np.testing.assert_allclose(d_vec_iterative, d_vecs[..., idx], rtol=0, atol=1e-10)

    print("Done")

    # d_vecs_iterative = grad_eigenvectors_hermitian_iterative(
    #     A, dA, vals=vals, vecs=vecs, d_vals=d_vals, offset=offset, verbose=True
    # )
    # np.testing.assert_allclose(d_vecs_iterative, d_vecs, rtol=0, atol=1e-10)
