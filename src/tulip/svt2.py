import numpy as np
import scipy.sparse as scis
from unipy import linalg, matmul, multiply

class SVT2:
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
        a: scis.dok_array,
        b: scis.csc_array,
        verbose: bool = False,
        **kwargs,
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
        self.m = self.a.shape[1]
        self.n = self.b.shape[1]
        self.kwargs = kwargs

    @property
    def c(self) -> np.array:
        """Reconstruct c (low-rank component) from svd components

        Returns
        -------
            array : reconstructed low-rank component

        """
        return matmul(multiply(self._c[0], self._c[1]), self._c[2])

    @c.setter
    def c(self, a) -> None:
        """Sets b from tuple

        Parameters
        ----------
        a : tuple[array]
            Tuple containing arrays of svd to set b

        """
        # print(type(self._b[1]))
        self._c[0] = a[0]
        self._c[1] = a[1]
        self._c[2] = a[2]
        return None

    def _initialize_c(self) -> None:
        """Initialize c"""
        self._c = [0, 0, 0]
        self._c[0] = np.zeros((self.m, 1), dtype=np.float32)
        self._c[1] = np.zeros((1, 1), dtype=np.float32)
        self._c[2] = np.zeros((1, self.n), dtype=np.float32)
        return None

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
        self._initialize_c()
        b_fro = scis.linalg.norm(self.b, "fro")
        # k0 = int(tau / delta / b_fro)
        Y = np.zeros(self.b.shape)
        k = 0
        pinv_a = np.linalg.pinv(self.a.toarray())
        self._sv = 10  # set sv
        self._svp = 0
        
        # Main SVT loop
        while k < k_max:
            # s = r + 1
            # Calculate SVD and perform soft thresholding
            self.kwargs.update({"sv": self._sv})
            U, S, Vt = linalg.svd(pinv_a @ Y, self.kwargs)
            self._svp = int(sum(S > tau))
            # Compute solution based on thresholded singular values
            if self._svp == 0:
                # self.c = U @ Vt
                self.c = (U, np.array([0]), Vt)
            else:
                self.c = (U[:, :self._svp], (S[:self._svp] - tau), Vt[:self._svp, :])
                # self.c = (U[:, :r] * (S[:r] - tau)) @ Vt[:r, :]
            self._update_sv() # update svps and svs
            # Check stopping criterion
            crit = linalg.norm((self.b - self.a @ self.c), "fro") / b_fro
            if crit < 1e-4:
                if self.verbose:
                    print(
                        f"Converged at iteration {k} - Final error = {crit:.6f}"
                    )
                break
            elif self.verbose:
                print(f"Iteration {k} - Error = {crit:.6f} - Rank = {self._svp}")

            # Update Y
            Y += delta * (self.b - self.a @ self.c)

            # Increment iteration counter
            k += 1

        if self.verbose and k == k_max:
            print(
                f"Maximum iterations ({k_max}) reached - Final error = {crit:.6f}"
            )
            
    def _update_sv(self) -> None:
        """Update (predict) the number of singular values to be calculated by svd."""
        n = min(self.m, self.n)
        if self._svp < self._sv:
            self._sv = int(min(self._svp + 1, n))
        else:
            self._sv = int(min(self._svp + 10, n))