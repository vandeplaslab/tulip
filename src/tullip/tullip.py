from typing import Union

import numpy as np
import scipy.sparse as scis


class TULLIP:
    """
    TULLIP (specTrometric Unmixing of single-ceLL by Inverse Problem) algorithm
    for matrix factorization with sparsity and non-negativity constraints.

    Parameters
    ----------
    a : scipy.sparse.dok_matrix or numpy.ndarray
        Measurement matrix A in the TULLIP problem.
    b : numpy.ndarray
        Observation matrix B in the TULLIP problem.
    omega : numpy.ndarray
        Binary mask matrix indicating observed entries.
    verbose : bool, optional
        Flag to enable verbose output, by default False.

    Attributes
    ----------
    a : scipy.sparse.dok_matrix or numpy.ndarray
        Measurement matrix A in the TULLIP problem.
    b : numpy.ndarray
        Observation matrix B in the TULLIP problem.
    omega : numpy.ndarray
        Binary mask matrix indicating observed entries.
    verbose : bool
        Flag to enable verbose output.
    h : numpy.ndarray
        Left factor matrix.
    w :  numpy.ndarray
        Right factor matrix.
    """

    def __init__(
        self,
        a: Union[scis.dok_matrix, np.ndarray],
        b: np.ndarray,
        omega: np.ndarray,
        verbose: bool = False,
    ):
        """
        Initialize the TULLIP class with the given parameters.

        Parameters:
        a (Union[scipy.sparse.dok_matrix, numpy.ndarray]): Measurement matrix A.
        b (numpy.ndarray): Observation matrix B.
        omega (numpy.ndarray): Binary mask matrix indicating observed entries.
        verbose (bool, optional): Flag to enable verbose output, by default False.
        """
        self.a = a
        self.b = b
        self.omega = omega
        self.verbose = verbose
        self.h = None
        self.w = None
        self.m = self.a.shape[1]
        self.n = self.b.shape[1]

    @property
    def c(self) -> np.ndarray:
        """
        Get the factorized matrix C = L @ R.

        Returns:
        numpy.ndarray: Factorized matrix.
        """
        if self.h is None or self.w is None:
            return None
        return self.h @ self.w

    def get_measurement_reconstruction(self) -> np.ndarray:
        """
        Get the full reconstruction including measurement matrix: A @ L @ R.

        Returns:
        numpy.ndarray: Reconstructed measurement matrix.
        """
        if self.h is None or self.w is None:
            return None
        return self.a @ self.h @ self.w

    @staticmethod
    def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply soft-thresholding to enforce sparsity.

        Parameters:
        x (numpy.ndarray): Input matrix.
        threshold (float): Threshold value.

        Returns:
        numpy.ndarray: Soft-thresholded matrix.
        """
        return np.maximum(x - threshold, 0)

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        """
        Normalize rows of matrix to sum to 1.

        Parameters:
        x (numpy.ndarray): Input matrix.

        Returns:
        numpy.ndarray: Row-normalized matrix.
        """
        row_sums = x.sum(axis=1, keepdims=True)
        return x / np.maximum(row_sums, 1e-10)

    def run(
        self,
        rank: int,
        sparsity: float = 0.1,
        max_iter: int = 100,
        tol: float = 1e-4,
        eta_l: float = 0.001,
        eta_r: float = 0.001,
        normalize: bool = False,
    ) -> None:
        """
        Run the TULLIP algorithm for matrix factorization.

        Parameters:
        rank (int): Rank of the factorization.
        sparsity (float, optional): Sparsity parameter for soft thresholding, by default 0.1.
        max_iter (int, optional): Maximum number of iterations, by default 100.
        tol (float, optional): Convergence tolerance, by default 1e-4.
        eta_l (float, optional): Step size for L update, by default 0.001.
        eta_r (float, optional): Step size for R update, by default 0.001.
        normalize (bool, optional): Whether to normalize rows of R, by default False.

        Returns:
        None
        """
        # Initialize factors with non-negative random values
        self.h = np.random.rand(self.m, rank)
        self.w = np.random.rand(rank, self.n)
        l_prev = self.h.copy()
        r_prev = self.w.copy()

        for iteration in range(max_iter):
            # Update L
            grad_l = (
                self.a.T
                @ (self.omega * ((self.a @ self.h) @ self.w - self.b))
                @ self.w.T
            ) + self.h
            self.h = self.h - eta_l * grad_l
            self.h = np.maximum(self.h, np.finfo("float32").eps)

            # Update R
            grad_r = (
                (self.a @ self.h).T
                @ (self.omega * ((self.a @ self.h) @ self.w - self.b))
            ) + self.w
            self.w = self.w - eta_r * grad_r
            self.w = np.maximum(self.w, np.finfo("float32").eps)
            self.w = self._soft_threshold(self.w, sparsity)

            if normalize:
                self.w = self._normalize_rows(self.w)

            # Check convergence
            l_diff = np.linalg.norm(self.h - l_prev, "fro")
            r_diff = np.linalg.norm(self.w - r_prev, "fro")

            if self.verbose:
                print(
                    f"Iteration {iteration + 1}: L diff = {l_diff:.6f}, R diff = {r_diff:.6f}"
                )

            if iteration > 0 and l_diff < tol and r_diff < tol:
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            l_prev, r_prev = self.h.copy(), self.w.copy()

        if self.verbose and iteration == max_iter - 1:
            print(f"Maximum iterations ({max_iter}) reached")
