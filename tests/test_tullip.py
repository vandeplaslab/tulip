import numpy as np
import pytest
import scipy.sparse as scis

from tullip.tullip import TULLIP


def test_tullip_initialization():
    # Test matrices and mask with both sparse and dense options for `a`
    a = scis.dok_matrix((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Initialize the TULLIP object
    tullip = TULLIP(a, b, omega, verbose=False)

    # Check that parameters are initialized correctly
    assert tullip.a is a
    assert np.array_equal(tullip.b, b)
    assert np.array_equal(tullip.omega, omega)
    assert tullip.verbose is False
    assert tullip.h is None
    assert tullip.w is None
    assert tullip.m == 3
    assert tullip.n == 3


def test_tullip_run_basic_factorization():
    # Initialize a small test setup
    a = scis.dok_matrix((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Run TULLIP
    tullip = TULLIP(a, b, omega, verbose=False)
    tullip.run(rank=2, max_iter=10)

    # Check that factor matrices `h` and `w` are initialized
    assert (
        tullip.h is not None
    ), "Matrix `h` should be initialized after running TULLIP"
    assert (
        tullip.w is not None
    ), "Matrix `w` should be initialized after running TULLIP"

    # Verify that the factorization result `c` has the correct shape
    assert tullip.c.shape == (3, 3), "`c` should have the same shape as `b`"


def test_tullip_run_with_sparsity():
    # Set up test case with sparse `a`
    a = scis.dok_matrix((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Run TULLIP with high sparsity threshold
    tullip = TULLIP(a, b, omega, verbose=False)
    tullip.run(rank=2, sparsity=0.5, max_iter=10)

    # Check if `w` contains zeros, indicating sparsity was enforced
    assert np.any(
        tullip.w == 0
    ), "Sparsity constraint not applied to matrix `w`"


def test_tullip_run_with_normalization():
    # Test normalization behavior
    a = scis.dok_matrix((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Run TULLIP with normalization enabled
    tullip = TULLIP(a, b, omega, verbose=False)
    tullip.run(rank=2, max_iter=10, normalize=True)

    # Check if rows of `w` are normalized
    row_sums = tullip.w.sum(axis=1)
    assert np.allclose(row_sums, 1), "Rows of `w` should be normalized"


@pytest.mark.parametrize(
    "rank, max_iter, tol",
    [
        (1, 5, 1e-2),
        (2, 10, 1e-3),
        (3, 15, 1e-4),
    ],
)
def test_tullip_run_with_varied_params(rank, max_iter, tol):
    # Test TULLIP with varied parameters
    a = scis.dok_matrix((3, 3), dtype=np.float32)
    a[0, 0], a[1, 1], a[2, 2] = 1, 1, 1
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    tullip = TULLIP(a, b, omega, verbose=False)
    tullip.run(rank=rank, max_iter=max_iter, tol=tol)

    # Verify that `h` and `w` matrices were generated and have expected shapes
    assert tullip.h.shape == (3, rank), f"`h` shape should be (3, {rank})"
    assert tullip.w.shape == (rank, 3), f"`w` shape should be ({rank}, 3)"
