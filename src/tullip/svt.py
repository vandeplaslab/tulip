import numpy as np
import scipy.sparse as scis


class SVT:
    """
    Perform matrix completion using Singular Value Thresholding (SVT).

    Parameters
    ----------
    a : scipy.sparse.dok_matrix
        Measurement matrix A in the SVT problem.
    b : numpy.ndarray
        Observation matrix Bin the SVT problem.
    verbose : bool, optional
        Flag to enable verbose output, by default False.

    Attributes
    ----------
    a : scipy.sparse.dok_matrix
        Measurement matrix A in the SVT problem.
    b : numpy.ndarray
        Observation matrix B in the SVT problem.
    verbose : bool
        Flag to enable verbose output.
    c : numpy.ndarray
        The completed matrix solution.
    """

    def __init__(
        self,
        a: scis.dok_matrix,
        b: np.ndarray,
        verbose: bool = False,
    ):
        """
        Initialize the SVT class with the given parameters.

        Parameters:
        a (scipy.sparse.dok_matrix): Measurement matrix A in the SVT problem.
        b (numpy.ndarray): Observation matrix B in the SVT problem.
        verbose (bool, optional): Flag to enable verbose output, by default False.
        """
        self.a = a
        self.b = b
        self.verbose = verbose
        self.c = None
        self.m = self.a.shape[1]
        self.n = self.b.shape[1]

    def run(
        self,
        k_max: int = 1000,
        delta: float = None,
        tau: float = None,
    ) -> None:
        """
        Compute the matrix completion solution using Singular Value Thresholding.

        Parameters:
        k_max (int, optional): Maximum number of iterations, by default 1000.
        delta (float, optional): Step size parameter, by default
                computed based on matrix properties.
        tau (float, optional): Threshold parameter, by default
                computed based on matrix properties.

        Returns:
        None
        """
        # Set default parameters if not provided
        if delta is None:
            delta = 1.2 * np.prod(self.b.shape) / self.b.nnz
        if tau is None:
            tau = self.b.shape[0]

        # Initialize variables
        b_fro = np.linalg.norm(self.b, "fro")
        # k0 = int(tau / delta / b_fro)
        Y = np.zeros_like(self.b)
        r = 0
        k = 0
        pinv_a = np.linalg.pinv(self.a)

        # Main SVT loop
        while k < k_max:
            # s = r + 1

            # Calculate SVD and perform soft thresholding
            U, S, Vt = np.linalg.svd(pinv_a @ Y, full_matrices=False)
            r = int(sum(S > tau))

            # Compute solution based on thresholded singular values
            if r == 0:
                self.c = U @ Vt
            else:
                self.c = (U[:, :r] * (S[:r] - tau)) @ Vt[:r, :]

            # Check stopping criterion
            crit = np.linalg.norm((self.b - self.a @ self.c), "fro") / b_fro
            if crit < 1e-4:
                if self.verbose:
                    print(
                        f"Converged at iteration {k} - Final error = {crit:.6f}"
                    )
                break
            elif self.verbose:
                print(f"Iteration {k} - Error = {crit:.6f} - Rank = {r}")

            # Update Y
            Y += delta * (self.b - self.a @ self.c)

            # Increment iteration counter
            k += 1

        if self.verbose and k == k_max:
            print(
                f"Maximum iterations ({k_max}) reached - Final error = {crit:.6f}"
            )
