# from typing import Union

import numpy as np
import scipy.sparse as scis
from numba import njit, prange

class TULIP2:
    def __init__(self, a, b, verbose=False):
        self.a = a # should be DOK
        self.b = b # should be CSC
        self.verbose = verbose
        self.h = None
        self.w = None
        self.m = self.a.shape[1]
        self.n = self.b.shape[1]
        self.best_h = None
        self.best_w = None
        self.best_loss = float("inf")

    def _adam_update(
        self, param, grad, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        """
        Adam optimization algorithm for adaptive learning rates

        Parameters:
        param (np.ndarray): Current parameter matrix
        grad (np.ndarray): Gradient of the parameter
        m (np.ndarray): First moment estimate
        v (np.ndarray): Second moment estimate
        t (int): Timestep

        Returns:
        tuple: Updated parameter, first moment, second moment
        """
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        # Parameter update
        param_updated = param - (m_hat / (np.sqrt(v_hat) + epsilon))
        return np.maximum(param_updated, np.finfo("float32").eps), m, v

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
        return (self.a @ self.h) @ self.w

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
        max_iter: int = 200,
        tol: float = 1e-5,
        restart_attempts: int = 3,
        normalize: bool = False,
    ) -> None:
        for attempt in range(restart_attempts):
            # Initialize factors with non-negative random values
            np.random.seed(42)
            self.h = np.random.rand(self.m, rank).astype(np.float32)
            np.random.seed(42)
            self.w = np.random.rand(rank, self.n).astype(np.float32)

            # Initialize Adam moments
            m_h = np.zeros_like(self.h)
            v_h = np.zeros_like(self.h)
            m_w = np.zeros_like(self.w)
            v_w = np.zeros_like(self.w)
            
            self.error = scis.csc_array(
                (np.zeros_like(self.b.data), self.b.indices, self.b.indptr), shape=self.b.shape
            )
            l_prev = self.h.copy()
            r_prev = self.w.copy()
    
            error_indices = self.error.indices
            error_indptr = self.error.indptr
            error_data = self.error.data.copy()  # Make a copy to modify
            b_data = self.b.data

            for iteration in range(max_iter):
                # Compute reconstruction error
                #self.sparse_error_calculation(self.h, self.w)
                error_data = sparse_error_calculation(self.a.toarray(), self.h, self.w, error_indices, error_indptr, error_data, self.error.shape, b_data)
                self.error.data = error_data
                
                # Update L (left factor)
                grad_l = (self.a.T @ self.error @ self.w.T) + self.h
                self.h, m_h, v_h = self._adam_update(
                    self.h, grad_l, m_h, v_h, iteration + 1
                )
                self.h = np.maximum(self.h, np.finfo("float32").eps)

                # Update R (right factor)
                grad_r = ((self.a @ self.h).T @ self.error) + self.w
                self.w, m_w, v_w = self._adam_update(
                    self.w, grad_r, m_w, v_w, iteration + 1
                )
                self.w = np.maximum(self.w, np.finfo("float32").eps)
                self.w = self._soft_threshold(self.w, sparsity)

                if normalize:
                    self.w = self._normalize_rows(self.w)

                # Compute current loss
                current_loss = (
                    np.linalg.norm(self.error.data, 2)
                    / np.linalg.norm(self.b.data, 2)
                    * 100
                )

                # Update best solution if current loss is lower
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_h = self.h.copy()
                    self.best_w = self.w.copy()

                # Check convergence
                l_diff = (
                    np.linalg.norm(self.h - l_prev, "fro")
                    / np.linalg.norm(l_prev, "fro")
                    * 100
                )
                r_diff = (
                    np.linalg.norm(self.w - r_prev, "fro")
                    / np.linalg.norm(r_prev, "fro")
                    * 100
                )

                if self.verbose:
                    print(
                        f"Attempt {attempt+1}, Iteration {iteration + 1}: "
                        f"Loss = {current_loss:.6f}, "
                        f"L diff = {l_diff:.6f}, R diff = {r_diff:.6f}"
                    )

                if iteration > 0 and l_diff < tol and r_diff < tol:
                    break

                l_prev, r_prev = self.h.copy(), self.w.copy()

            # If no improvement, continue to next restart
            if self.best_h is not None:
                self.h = self.best_h
                self.w = self.best_w
                break

        if self.verbose:
            print(f"Final Loss: {self.best_loss}")
            
#     @njit
#     def sparse_error_calculation(self, W, H):
#         """
#         Compute a partial matrix product with sparse result 
#         constrained by non-zero pattern of input sparse matrix B and subtract its values from it

#         Parameters:
#         -----------
#         W : numpy array or sparse matrix
#         H : numpy array or sparse matrix
#         """
#         # Compute partial intermediate product
#         intermediate = self.a @ W

#         # Subtract corresponding values from B
#         for i in range(self.error.shape[1]):
#             col = i
#             row = self.error.indices[self.error.indptr[i] : self.error.indptr[i + 1]]
#             self.error.data[self.error.indptr[i] : self.error.indptr[i + 1]] = (
#                 intermediate[row, :] @ H[:, col]
#             )
        
#         self.error.data -= self.b.data

#         return None

@njit(parallel=True, fastmath=True)
def sparse_error_calculation(a, W, H, error_indices, error_indptr, error_data, error_shape, b_data):
    # Rewrite the method using Numba-compatible constructs
    intermediate = a @ W

    # Use prange instead of range for potential parallelization
    for i in prange(error_shape[1]):
        col = i
        row_start = error_indptr[i]
        row_end = error_indptr[i + 1]
        rows = error_indices[row_start:row_end]

        error_data[row_start:row_end] = (
            intermediate[rows, :] @ np.ascontiguousarray(H[:, col])
        )
        
    error_data -= b_data

    return error_data