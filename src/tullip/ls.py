from typing import Tuple

import numpy as np
import scipy.sparse as scis


class LS:
    """
    Perform unmixing methods using the least-squares approach.

    Parameters
    ----------
    a : scipy.sparse.dok_matrix
        Matrix A in the least-squares problem.
    b : numpy.ndarray
        Vector b in the least-squares problem.
    lambda_factor : float, optional
        Regularization factor for the least-squares solution, by default 0.

    Attributes
    ----------
    a : scipy.sparse.dok_matrix
        Matrix A in the least-squares problem.
    b : numpy.ndarray
        Vector b in the least-squares problem.
    lambda_factor : float
        Regularization factor for the least-squares solution.
    c : numpy.ndarray
        The least-squares solution.
    """

    def __init__(
        self,
        a: scis.dok_matrix,
        b: np.ndarray,
        lambda_factor: float = 0,
    ):
        """
        Initialize the LS class with the given parameters.

        Parameters:
        a (scipy.sparse.dok_matrix): Matrix A in the least-squares problem.
        b (numpy.ndarray): Vector b in the least-squares problem.
        lambda_factor (float, optional): Regularization factor for the
                                    least-squares solution, by default 0.
        """
        self.a = a
        self.b = b
        self.lambda_factor = lambda_factor
        self.c = None

    def run(self) -> None:
        """
        Compute the least-squares solution and store it in the `c` attribute.

        Returns:
        None
        """
        self.c = scis.linalg.splu(
            (
                self.a.transpose() @ self.a
                + self.lambda_factor * scis.eye(self.a.shape[1])
            ).tocsc()
        )
        self.c = self.c.solve(self.a.transpose() @ self.b)
