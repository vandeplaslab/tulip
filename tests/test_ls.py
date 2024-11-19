import numpy as np
import pytest
import scipy.sparse as scis

from tulip.ls import LS


@pytest.fixture
def small_problem():
    """Fixture for a small least squares problem with known solution."""
    # Create a small, well-conditioned problem
    a = scis.dok_matrix((3, 2))
    a[0, 0] = 1
    a[1, 0] = 2
    a[2, 0] = 3
    a[0, 1] = 4
    a[1, 1] = 5
    a[2, 1] = 6

    # Known solution c = [1, 1]
    b = np.array([5, 7, 9])
    return a, b


@pytest.fixture
def large_problem():
    """Fixture for a larger least squares problem."""
    n, m = 100, 50
    # Create random sparse matrix
    density = 0.1
    a = scis.random(n, m, density=density, format="dok")
    # Create random solution
    true_c = np.random.randn(m)
    # Generate corresponding b with some noise
    b = a @ true_c + np.random.normal(0, 0.01, n)
    return a, b, true_c


def test_initialization():
    """Test proper initialization of LS class."""
    a = scis.dok_matrix((3, 2))
    b = np.array([1, 2, 3])
    lambda_factor = 0.1

    ls = LS(a, b, lambda_factor)

    assert isinstance(ls.a, scis.dok_matrix)
    assert isinstance(ls.b, np.ndarray)
    assert ls.lambda_factor == 0.1
    assert ls.c is None


def test_small_problem_solution(small_problem):
    """Test solution for small problem with known result."""
    a, b = small_problem
    ls = LS(a, b)
    ls.run()

    # Expected solution should be close to [1, 1]
    expected = np.array([1.0, 1.0])
    assert np.allclose(ls.c, expected, rtol=1e-10, atol=1e-10)


def test_large_problem_solution(large_problem):
    """Test solution for larger random problem."""
    a, b, true_c = large_problem
    ls = LS(a, b)
    ls.run()

    # Check if the solution minimizes the least squares error
    residual = np.linalg.norm(a @ ls.c - b)
    # Check if slightly perturbed solutions give larger residuals
    perturbation = np.random.normal(0, 0.01, len(ls.c))
    perturbed_residual = np.linalg.norm(a @ (ls.c + perturbation) - b)

    assert residual < perturbed_residual


def test_regularization():
    """Test if regularization parameter affects the solution."""
    # Create a well-conditioned but ill-scaled problem
    a = scis.dok_matrix((4, 2))
    a[0, 0] = 1.0
    a[1, 0] = 2.0
    a[2, 0] = 0.1
    a[3, 0] = 0.2
    a[0, 1] = 0.1
    a[1, 1] = 0.2
    a[2, 1] = 1.0
    a[3, 1] = 2.0

    # Create a corresponding b vector
    b = np.array([1.0, 2.0, 1.0, 2.0])

    # Solve with and without regularization
    ls_no_reg = LS(a, b, lambda_factor=0)
    ls_with_reg = LS(a, b, lambda_factor=1.0)

    ls_no_reg.run()
    ls_with_reg.run()

    # Solutions should be different with regularization
    assert not np.allclose(ls_no_reg.c, ls_with_reg.c)

    # Both solutions should produce reasonable residuals
    residual_no_reg = np.linalg.norm(a @ ls_no_reg.c - b)
    residual_with_reg = np.linalg.norm(a @ ls_with_reg.c - b)

    # Regularized solution should have smaller norm
    assert np.linalg.norm(ls_with_reg.c) < np.linalg.norm(ls_no_reg.c)

    # Unregularized solution should have smaller residual
    assert residual_no_reg <= residual_with_reg


# def test_input_validation():
#     """Test if initialization fails with invalid inputs."""
#     with pytest.raises((ValueError, TypeError)):
#         # Try with incompatible dimensions
#         a = scis.dok_matrix((3, 2))
#         b = np.array([1, 2])  # Should be length 3
#         LS(a, b)

#     with pytest.raises((ValueError, TypeError)):
#         # Try with wrong matrix dimensions for multiplication
#         a = scis.dok_matrix((2, 3))
#         b = np.array([1, 2, 3])  # Dimensions don't match for A^T @ A
#         ls = LS(a, b)
#         ls.run()


def test_solution_shape():
    """Test if solution has correct shape."""
    # Create a well-conditioned overdetermined system
    a = scis.dok_matrix((6, 3))
    # Make sure the matrix is well-conditioned
    a[0, 0] = 1.0
    a[1, 0] = 0.5
    a[2, 1] = 1.0
    a[3, 1] = 0.5
    a[4, 2] = 1.0
    a[5, 2] = 0.5

    b = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5])

    ls = LS(a, b)
    ls.run()

    assert ls.c.shape == (3,)


def test_zero_regularization():
    """Test if zero regularization gives same result as no regularization."""
    a = scis.dok_matrix((4, 2))
    a[0, 0] = 1.0
    a[1, 1] = 1.0
    a[2, 0] = 0.5
    a[3, 1] = 0.5
    b = np.array([1.0, 1.0, 0.5, 0.5])

    ls1 = LS(a, b)  # default lambda_factor=0
    ls2 = LS(a, b, lambda_factor=0.0)  # explicit zero

    ls1.run()
    ls2.run()

    assert np.allclose(ls1.c, ls2.c)
