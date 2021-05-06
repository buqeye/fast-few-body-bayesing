import pytest
from fit3bf import Hamiltonian, Operator
from fit3bf.test_utils import create_random_psd_matrix
import pickle

import numpy as np


def test_hamiltonian_serialization():
    # Create test observable
    n = 10
    n_p = 5
    H0 = create_random_psd_matrix(n, random_state=1)
    H1 = np.stack([
        create_random_psd_matrix(n, random_state=i) for i in range(n_p)
    ], axis=-1)
    p_train = np.array([np.arange(i, n_p+i) for i in range(13)])
    op = Hamiltonian('test', H0=H0, H1=H1)
    op.fit_evc(p_train)

    d = op.__dict__
    # Simulate pickling and un-pickling
    op_new = pickle.loads(pickle.dumps(op))
    d_new = op_new.__dict__

    # Test that all keys are identical except those in _large_attrs
    for key, val in d_new.items():
        if key in op._large_attrs:
            assert val is None
        else:
            if isinstance(val, np.ndarray):
                np.testing.assert_allclose(val, d[key])
            else:
                assert val == d[key]
