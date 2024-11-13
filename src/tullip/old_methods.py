from typing import Tuple, Union

import cvxpy as cp
import numpy as np
import scipy.sparse as ss
from joblib import Parallel, delayed
from scipy.optimize import nnls


class unmix:
    """
    Perform unmixing methods


    Parameters
    ----------

    Attributes
    ----------


    """

    def __init__(
        self,
    ):
        self.c = None
        pass

    def solve_ls(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        lambda_factor: float = 0,
    ) -> None:
        self.c = ss.linalg.splu(
            (a.transpose() @ a + lambda_factor * ss.eye(a.shape[1])).tocsc()
        )
        self.c = self.c.solve(a.transpose() @ b)

        return None

    def solve_nnls(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        verbose: bool = False,
    ):
        m = a.shape[1]
        n = b.shape[1]
        c = cp.Variable((m, n))
        objective = cp.Minimize(cp.norm((a @ c - b), "fro"))
        constraints = [0 <= cp.vec(c)]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        self.c = c.value

        return None

    def solve_nnls_scipy(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        verbose: bool = False,
    ):
        m = a.shape[1]
        n = b.shape[1]
        c = np.zeros((m, n))
        a = a.todense()
        for i in range(n):
            c[:, i] = nnls(a, b[:, i])[0]
            if verbose:
                print(i, " from ", n)
        self.c = c

        return None

    def _nnls_o(self, a: ss.dok_matrix, b: np.ndarray):
        return nnls(a, b)[0]

    def solve_nnls_scipy_parallel(
        self, a: ss.dok_matrix, b: np.ndarray, n_jobs: int = 1
    ) -> None:
        m = a.shape[1]
        n = b.shape[1]
        c = np.zeros((m, n))
        a = a.todense()

        c = Parallel(n_jobs=n_jobs)(delayed(self._nnls_o)(a, b[:, i]) for i in range(n))
        self.c = np.array(c).T

        return None

    def solve_nnm(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        lambda_factor: float = 1,
        verbose: bool = False,
    ):
        m = a.shape[1]
        n = b.shape[1]
        c = cp.Variable((m, n))
        objective = cp.Minimize(
            cp.norm((a @ c - b), "fro") + lambda_factor * cp.norm(c, "nuc")
        )
        constraints = []
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        self.c = c.value

        return None

    def solve_nn_nnm(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        lambda_factor: float = 1,
        verbose: bool = False,
    ):
        m = a.shape[1]
        n = b.shape[1]
        c = cp.Variable((m, n))
        objective = cp.Minimize(
            cp.norm((a @ c - b), "fro") + lambda_factor * cp.norm(c, "nuc")
        )
        constraints = [0 <= cp.vec(c)]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        self.c = c.value

        return None

    def solve_lr(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        lambda_factor: float = 1000,
        rank: int = 1,
        verbose: bool = False,
    ):
        m = a.shape[1]
        n = b.shape[1]
        l = cp.Variable((m, rank))
        r = cp.Variable((rank, n))
        objective = cp.Minimize(0.5 * (cp.norm(l, "fro") ** 2 + cp.norm(r, "fro") ** 2))
        constraints = [
            cp.norm(a @ l @ r - b, "fro") <= lambda_factor,
            0 <= cp.vec(l),
            0 <= cp.vec(r),
        ]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=verbose)
        self.c = c.value

        return None

    def solve_fcnnls(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        verbose: bool = False,
    ):
        def cssls(CtC, CtA, Pset=None):
            # Solve the set of equations CtA = CtC*K for the variables in set Pset
            # using the fast combinatorial approach

            Z = np.zeros_like(CtA)  # create solution

            if Pset is None or np.all(Pset):  # for first iteration
                Z = np.linalg.solve(CtC, CtA)
            else:
                lVar, pRHS = Pset.shape
                codedPset = (2 ** np.arange(lVar - 1, -1, -1)) @ Pset

                sorted_indices = np.argsort(codedPset)
                sortedPset = codedPset[sorted_indices]
                sortedEset = np.arange(pRHS)[sorted_indices]
                breaks = np.diff(sortedPset)

                breakIdx = np.concatenate(([0], np.where(breaks)[0] + 1, [pRHS]))
                for k in range(len(breakIdx) - 1):
                    cols2solve = sortedEset[breakIdx[k] : breakIdx[k + 1]]
                    varsi = Pset[:, sortedEset[breakIdx[k]]]
                    Z[np.ix_(varsi, cols2solve)] = np.linalg.solve(
                        CtC[np.ix_(varsi, varsi)], CtA[np.ix_(varsi, cols2solve)]
                    )
            return Z

        def fcnnls(C: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            # NNLS using normal equations and the fast combinatorial strategy
            #
            # I/O: [K, Pset] = fcnnls(C, A);
            # K = fcnnls(C, A);
            #
            # C is the nObs x lVar coefficient matrix
            # A is the nObs x pRHS matrix of observations
            # K is the lVar x pRHS solution matrix
            # Pset is the lVar x pRHS passive set logical array

            # Check the input arguments for consistency and initialize
            if A.shape[0] != C.shape[0]:
                raise ValueError("C and A have incompatible sizes")

            # Initialize some variables
            nObs, lVar = C.shape
            pRHS = A.shape[1]
            W = np.zeros((lVar, pRHS))
            i = 0
            maxiter = 3 * lVar
            maxiter_e = 100
            # Initialize some variables

            # Precompute parts of pseudoinverse
            CtC = C.T @ C
            CtA = C.T @ A
            # Precompute parts of pseudoinverse

            # Obtain the initial feasible solution and corresponding passive set
            K = cssls(CtC, CtA)  # pseudo inverse
            Pset = K > 0  # Passive set is those where K already fulfills conditions
            K[
                ~Pset
            ] = 0  # Set all values that are not in the passive set to 0 (comparable to NMF)
            D = K  # Create a copy of K
            Fset = np.where(~np.all(Pset, axis=0))[
                0
            ]  # Find the columns that have some active members
            # Obtain the initial feasible solution and corresponding passive (Pset) and active set(Fset)

            e = 0
            while (
                Fset.any() and e < maxiter_e
            ):  # while there are still columns with not all feasible solutions
                # Retry LS Solve for the columns that still have infeasible solutions
                K[:, Fset] = cssls(CtC, CtA[:, Fset], Pset[:, Fset])

                # Find any infeasible solutions
                Hset = np.where(np.any(K[:, Fset] < 0, axis=0))[
                    0
                ]  # FInd in our new solution possible negative values
                if Hset.size > 0:
                    i = 0
                    alpha = np.zeros((lVar, Hset.size))
                    # Make infeasible solutions feasible (standard NNLS inner loop)
                    while Hset.size > 0 and i < maxiter:
                        i += 1
                        alpha[:, : Hset.size] = np.inf

                        xi, xj = np.where(Pset[:, Hset] & (K[:, Hset] < 0))

                        alpha[xi, xj] = D[xi, Hset[xj]] / (
                            D[xi, Hset[xj]] - K[xi, Hset[xj]]
                        )
                        alphaMin = np.nanmin(alpha[:, : Hset.size], axis=0)
                        minIdx = np.nanargmin(alpha[:, : Hset.size], axis=0)
                        alpha = np.tile(alphaMin, (lVar, 1))
                        D[:, Hset] = D[:, Hset] - alpha * (D[:, Hset] - K[:, Hset])

                        D[minIdx, Hset] = 0
                        Pset[minIdx, Hset] = 0
                        K[:, Hset] = cssls(CtC, CtA[:, Hset], Pset[:, Hset])
                        Hset = np.where(np.any(K < 0, axis=0))[0]

                # Make sure the solution has converged
                if i == maxiter:
                    raise ValueError("Maximum number iterations exceeded")

                # Check solutions for optimality
                W[:, Fset] = CtA[:, Fset] - np.matmul(CtC, K[:, Fset])
                Jset = np.where(np.all(~(Pset[:, Fset]) * W[:, Fset] <= 0, axis=0))[0]
                Fset = np.setdiff1d(Fset, Fset[Jset])

                # For non-optimal solutions, add the appropriate variable to Pset
                if len(Fset) != 0:
                    # mx = np.max(~(Pset[:, Fset]) * W[:, Fset], axis=0)
                    mxidx = np.argmin(~(Pset[:, Fset]) * W[:, Fset], axis=0)
                    Pset[mxidx, Fset] = 1
                    D[:, Fset] = K[:, Fset]

                print(Fset.any(), Fset.size, e)
                e += 1

            return K, Pset

        self.c, _ = fcnnls(b, a)

        return None

    def nnls_pgd(
        self,
        a: ss.dok_matrix,
        b: np.ndarray,
        maxiter: int = 100,
        verbose: bool = False,
    ):
        P = a.T @ b
        Q = a.T @ a
        t = 1 / ss.linalg.norm(Q, "fro")
        theta = t * P
        Theta = ss.eye(Q.shape[0]) - t * Q
        k = 0
        eps = np.finfo(np.float64).eps
        x = np.random.rand(P.shape[0], P.shape[1])
        while k < maxiter:
            x_k = np.clip(Theta @ x + theta, eps, None)
            x = (x_k).astype(np.float64, copy=True)
            k = k + 1
            if verbose:
                print(k, np.linalg.norm(a @ x - b, "fro"))

        self.c = x

        return None

    def nnls_admm(
        self,
        A,
        Y,
        rho=0,
        k=100,
        verbose: bool = False,
    ):
        # Predefine variables
        X = np.zeros((A.shape[1], Y.shape[1]))
        W = np.zeros((A.shape[1], Y.shape[1]))
        Lambd = np.zeros((A.shape[1], Y.shape[1]))

        Y_fro = np.linalg.norm(Y, "fro")

        if rho == 0:
            rho = np.linalg.norm(A, "fro")  # mistake here

        # Pre calculate:
        pinv = np.linalg.inv(
            np.array((A.T @ A + 1 * ss.eye(A.shape[1], dtype=np.float32)).todense())
        ).astype(np.float32)
        AtY = np.array(A.T @ Y)

        for i in range(k):
            X = pinv @ (AtY + rho * (Lambd - W))
            Lambd = np.clip(X + W, np.finfo(np.float32).eps, None)
            W += X - Lambd
            if verbose:
                print(i, np.linalg.norm(A @ X - Y, "fro") / Y_fro * 100)

        self.c = X

        return None
