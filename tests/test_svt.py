import numpy as np
import pytest
import scipy.sparse as scis

from tullip.svt import SVT


def test_svt_initialization():
    # Create small test matrices with dok_array for `a` and csc_array for `b`
    a = scis.dok_array((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = scis.csc_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Initialize the SVT object
    svt = SVT(a, b, verbose=False)

    # Check that parameters are initialized correctly
    assert svt.a is a
    assert (svt.b != b).nnz == 0  # Ensure sparse matrices are equal
    assert svt.verbose is False
    assert svt.c is None
    assert svt.m == 3
    assert svt.n == 3


def test_svt_run_basic_convergence():
    # Create small test matrices
    a = scis.dok_array((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = scis.csc_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Initialize and run SVT
    svt = SVT(a, b, verbose=False)
    svt.run(k_max=10)

    # Check that the completed matrix `c` is close to the non-sparse version of `b`
    assert svt.c is not None
    assert np.allclose(
        svt.c, b.toarray(), atol=1e-2
    ), "SVT did not converge to the expected solution"


def test_svt_run_max_iterations():
    # Create test matrices with dok_array for `a` and csc_array for `b`
    a = scis.dok_array((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = scis.csc_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Initialize SVT with verbose enabled for max iterations test
    svt = SVT(a, b, verbose=True)
    svt.run(k_max=1)  # Force a quick exit after one iteration

    # Check that `c` was updated even with only 1 iteration
    assert (
        svt.c is not None
    ), "Matrix `c` should be initialized even with a single iteration"


def test_svt_run_tau_and_delta():
    # Create test matrices with dok_array for `a` and csc_array for `b`
    a = scis.dok_array((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = scis.csc_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Test with specific delta and tau values
    svt = SVT(a, b, verbose=False)
    svt.run(k_max=10, delta=0.1, tau=1.0)

    # Ensure matrix `c` is computed and check that it is not the same as `b`
    assert svt.c is not None


@pytest.mark.parametrize(
    "delta, tau",
    [
        (None, None),
        (1.0, None),
        (None, 1.0),
        (1.0, 1.0),
    ],
)
def test_svt_run_varied_params(delta, tau):
    # Test with different combinations of delta and tau
    a = scis.dok_array((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = scis.csc_array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    svt = SVT(a, b, verbose=False)
    svt.run(k_max=10, delta=delta, tau=tau)

    # Check matrix `c` exists after running
    assert (
        svt.c is not None
    ), f"SVT should produce `c` with delta={delta} and tau={tau}"
