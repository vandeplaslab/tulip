# import cvxpy as cp
import numpy as np
import scipy.sparse as scis
from joblib import Parallel, delayed
from scipy.optimize import nnls


class NNLS:
    """
    Perform unmixing methods using the non-negative least-squares approach.

    Parameters
    ----------
    a : scipy.sparse.dok_matrix
        Matrix A in the non-negative least-squares problem.
    b : numpy.ndarray
        Matrix B in the non-negative least-squares problem.
    verbose : bool, optional
        Flag to enable verbose output, by default False.
    n_jobs : int, optional
        Number of parallel jobs to use for the non-negative least-squares solve,
            by default 1.

    Attributes
    ----------
    a : scipy.sparse.dok_matrix
        Matrix A in the non-negative least-squares problem.
    b : numpy.ndarray
        Matrix B in the non-negative least-squares problem.
    verbose : bool
        Flag to enable verbose output.
    n_jobs : int
        Number of parallel jobs to use for the non-negative least-squares solve.
    c : numpy.ndarray
        The non-negative least-squares solution.
    m : int
        Number of columns in matrix A.
    n : int
        Number of columns in matrix B.
    """

    def __init__(
        self,
        a: scis.dok_matrix,
        b: np.ndarray,
        verbose: bool = False,
        n_jobs: int = 1,
    ):
        """
        Initialize the NNLS class with the given parameters.

        Parameters:
        a (scipy.sparse.dok_matrix): Matrix A in the non-negative least-squares problem.
        b (numpy.ndarray): Matrix B in the non-negative least-squares problem.
        verbose (bool, optional): Flag to enable verbose output, by default False.
        n_jobs (int, optional): Number of parallel jobs to use for the non-negative
                            least-squares solve, by default 1.
        """
        self.a = a
        self.b = b
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.c = None
        self.m = self.a.shape[1]
        self.n = self.b.shape[1]

    # def run_cvxpy(self) -> None:
    #     """
    #     Compute the non-negative least-squares solution and store it
    #             in the `c` attribute.

    #     Returns:
    #     None
    #     """
    #     c = cp.Variable((self.m, self.n))
    #     objective = cp.Minimize(cp.norm((self.a @ c - self.b), "fro"))
    #     constraints = [0 <= cp.vec(c)]
    #     prob = cp.Problem(objective, constraints)
    #     prob.solve(verbose=self.verbose)
    #     self.c = c.value

    def run(self) -> None:
        """
        Compute the non-negative least-squares solution using SciPy's nnls
                function and store it in the `c` attribute.

        Returns:
        None
        """
        self.c = np.zeros((self.m, self.n))
        self.a = self.a.todense()
        for i in range(self.n):
            self.c[:, i] = nnls(self.a, self.b[:, i])[0]
            if self.verbose:
                print(i, " from ", self.n)

    def _nnls_o(self, a, b):
        """
        Helper function to compute the non-negative least-squares solution
                for a single column of B.

        Parameters:
        a (numpy.ndarray): Matrix A in the non-negative least-squares problem.
        b (numpy.ndarray): Vector b in the non-negative least-squares problem.

        Returns:
        numpy.ndarray: The non-negative least-squares solution.
        """
        return nnls(a, b)[0]

    def run_scipy_parallel(self) -> None:
        """
        Compute the non-negative least-squares solution using SciPy's nnls
                function in parallel and store it in the `c` attribute.

        Returns:
        None
        """
        self.c = np.zeros((self.m, self.n))
        self.a = self.a.todense()
        self.c = Parallel(n_jobs=self.n_jobs)(
            delayed(self._nnls_o)(self.a, self.b[:, i]) for i in range(self.n)
        )
        self.c = np.array(self.c).T

    def run_admm(
        self,
        rho: float = 0,
        k: int = 100,
    ) -> None:
        """
        Compute the non-negative least-squares solution using the Alternating
        Direction Method of Multipliers (ADMM) algorithm.

        Parameters:
        rho (float, optional): Penalty parameter for the ADMM algorithm, by default 0.
        k (int, optional): Number of ADMM iterations to perform, by default 100.

        Returns:
        None
        """
        # Predefine variables
        self.c = np.zeros((self.m, self.n))
        W = np.zeros((self.m, self.n))
        Lambd = np.zeros((self.m, self.n))
        b_fro = np.linalg.norm(self.b, "fro")

        # Set the value of rho if it's 0
        if rho == 0:
            rho = 1 / scis.linalg.norm(self.a, "fro")

        # Pre-calculate the pseudo-inverse of
        # (self.a.T @ self.a + 1 * scis.eye(self.m, dtype=np.float32))
        pinv = np.linalg.inv(
            np.array(
                (
                    self.a.T @ self.a + 1 * scis.eye(self.m, dtype=np.float32)
                ).todense()
            )
        ).astype(np.float32)
        AtY = np.array(self.a.T @ self.b)

        # Run the ADMM algorithm for k iterations
        for i in range(k):
            self.c = pinv @ (AtY + rho * (Lambd - W))
            Lambd = np.clip(self.c + W, np.finfo(np.float32).eps, None)
            W += self.c - Lambd

            # Print the current relative error if verbose is True
            if self.verbose:
                print(
                    i,
                    np.linalg.norm(self.a @ self.c - self.b, "fro")
                    / b_fro
                    * 100,
                )

        self.c = np.clip(self.c, np.finfo(np.float32).eps, None)
