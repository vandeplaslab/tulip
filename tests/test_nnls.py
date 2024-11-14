import numpy as np
import pytest
import scipy.sparse as scis

from tullip.nnls import NNLS


@pytest.fixture
def small_problem():
    """Fixture for a small non-negative least-squares problem."""
    # Create a small, well-conditioned problem
    a = scis.dok_matrix((3, 2))
    a[0, 0] = 1
    a[1, 0] = 2
    a[2, 0] = 3
    a[0, 1] = 4
    a[1, 1] = 5
    a[2, 1] = 6

    # Create a positive B matrix
    b = np.abs(np.random.randn(3, 3))
    return a, b


@pytest.fixture
def large_problem():
    """Fixture for a larger non-negative least-squares problem."""
    n, m = 100, 50
    # Create random sparse matrix A
    density = 0.1
    a = scis.random(n, m, density=density, format="dok")
    # Create random positive B matrix
    b = np.abs(np.random.randn(n, 30))
    return a, b


def test_small_problem_solution(small_problem):
    """Test solution for small problem."""
    a, b = small_problem
    nnls = NNLS(a, b)
    nnls.run()

    # Check if the solution is non-negative
    assert np.all(nnls.c >= 0)


def test_large_problem_solution(large_problem):
    """Test solution for larger problem."""
    a, b = large_problem
    nnls = NNLS(a, b)
    nnls.run()

    # Check if the solution is non-negative
    assert np.all(nnls.c >= 0)

    # Check if the solution minimizes the Frobenius norm
    residual = np.linalg.norm(a @ nnls.c - b, "fro")

    # Check if slightly perturbed solutions give larger residuals
    perturbation = np.abs(np.random.normal(0, 0.01, nnls.c.shape))
    perturbed_residual = np.linalg.norm(a @ (nnls.c + perturbation) - b, "fro")

    assert residual < perturbed_residual


# def test_scipy_solution(small_problem):
#     """Test solution using SciPy's nnls function."""
#     a, b = small_problem
#     nnls = NNLS(a, b)
#     nnls.run_scipy()

#     # Check if the solution is non-negative
#     assert np.all(nnls.c >= 0)


def test_scipy_parallel_solution(large_problem):
    """Test parallel solution using SciPy's nnls function."""
    a, b = large_problem
    nnls = NNLS(a, b, n_jobs=4)
    nnls.run_scipy_parallel()

    # Check if the solution is non-negative
    assert np.all(nnls.c >= 0)

    # Check if the solution minimizes the Frobenius norm
    residual = np.linalg.norm(a @ nnls.c - b, "fro")

    # Check if slightly perturbed solutions give larger residuals
    perturbation = np.abs(np.random.normal(0, 0.01, nnls.c.shape))
    perturbed_residual = np.linalg.norm(a @ (nnls.c + perturbation) - b, "fro")

    assert residual < perturbed_residual


def test_admm_solution(small_problem):
    """Test solution using the ADMM algorithm."""
    a, b = small_problem
    nnls = NNLS(a, b)
    nnls.run_admm()

    # Check if the solution is non-negative
    assert np.all(nnls.c >= 0)

    # Check if the solution minimizes the Frobenius norm
    residual = np.linalg.norm(a @ nnls.c - b, "fro")

    # Check if slightly perturbed solutions give larger residuals
    perturbation = np.abs(np.random.normal(0, 0.01, nnls.c.shape))
    perturbed_residual = np.linalg.norm(a @ (nnls.c + perturbation) - b, "fro")

    assert residual < perturbed_residual
