# from typing import Union

import numpy as np
import scipy.sparse as scis


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
            self.h = np.random.rand(self.m, rank)
            self.w = np.random.rand(rank, self.n)

            # Initialize Adam moments
            m_h = np.zeros_like(self.h)
            v_h = np.zeros_like(self.h)
            m_w = np.zeros_like(self.w)
            v_w = np.zeros_like(self.w)

            l_prev = self.h.copy()
            r_prev = self.w.copy()

            for iteration in range(max_iter):
                # Compute reconstruction error
                error = self.sparse_error_calculation(self.h, self.w)
                
                # Update L (left factor)
                grad_l = (self.a.T @ error @ self.w.T) + self.h
                self.h, m_h, v_h = self._adam_update(
                    self.h, grad_l, m_h, v_h, iteration + 1
                )
                self.h = np.maximum(self.h, np.finfo("float32").eps)

                # Update R (right factor)
                grad_r = ((self.a @ self.h).T @ error) + self.w
                self.w, m_w, v_w = self._adam_update(
                    self.w, grad_r, m_w, v_w, iteration + 1
                )
                self.w = np.maximum(self.w, np.finfo("float32").eps)
                self.w = self._soft_threshold(self.w, sparsity)

                if normalize:
                    self.w = self._normalize_rows(self.w)

                # Compute current loss
                current_loss = (
                    np.linalg.norm(error.data, 2)
                    / np.linalg.norm(self.b.data, 2)
                    * 100
                )
                test_loss = (
                    np.linalg.norm(error2.data, 2)
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
                        f"Loss = {test_loss:.6f}, "
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
            

    def sparse_error_calculation(self, W, H):
        """
        Compute a partial matrix product with sparse result 
        constrained by non-zero pattern of input sparse matrix B and subtract its values from it

        Parameters:
        -----------
        A : numpy array or sparse matrix
        W : numpy array or sparse matrix
        H : numpy array or sparse matrix
        B : scipy sparse CSC matrix (pattern matrix)

        Returns:
        --------
        C : scipy.sparse.csc_matrix with values only where B has non-zero entries
        """
        # Get row and column indices from B
        B_rows, B_cols = self.b.nonzero()

        # Compute partial intermediate product
        intermediate = self.a @ W

#         # Extract the partial result for B's non-zero indices
#         values = intermediate[B_rows, :] @ H[:, B_cols]
       
#         # Create a sparse selector matrix
#         selector = scis.csr_matrix(
#             (np.ones_like(B_rows), (np.arange(len(B_rows)), B_cols)), 
#             shape=(len(B_rows), H.shape[1])
#         )

#         # Compute values with minimal memory overhead
#         values = ((intermediate[B_rows, :] @ (H * selector.T)).diagonal()).copy()
        values = np.zeros(len(B_rows), dtype=float)

        for i, (row, col) in enumerate(zip(B_rows, B_cols)):
            values[i] = np.dot(intermediate[row, :], H[:, col])
        
        # Subtract corresponding values from B
        values -= self.b.data
        
        # Create sparse CSC matrix
        C = scis.csc_matrix(
            (values, (B_rows, B_cols)), 
            shape=self.b.shape
        )

        return C
