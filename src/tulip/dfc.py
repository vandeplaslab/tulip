import contextlib
import uuid

import joblib
import numpy as np
import scipy.sparse as scis
from joblib import Parallel, delayed

# from pyspa import SPAReader
from tqdm import tqdm

# from unipy import *

# from tulip.svt import SVT


class DFC:
    def __init__(
        self, reader, selection, svt, overlap, normalizer, save_path: str = ""
    ):
        self.reader = reader
        self.selection = selection
        self.svt = svt
        self.overlap = overlap
        self.normalizer = normalizer
        self.A = []
        self.save_path = save_path
        self.Uc = None
        self.Mc = None
        self.partition = {}

        return None

    def divide(self, bin_width: int = 100) -> None:
        # Create Subsampling
        vect = np.arange(len(self.selection))
        np.random.shuffle(vect)
        n_i = int(np.ceil(len(self.selection) / bin_width))
        for i in range(n_i - 1):
            self.partition[i] = vect[i * bin_width : (i + 1) * bin_width]

        self.partition[n_i - 1] = vect[(n_i - 1) * bin_width :]
        print(
            len(self.partition), "times", self.reader.n_mz_bins, "x", bin_width
        )

        return None

    def factor(self, n_jobs: int = 10):
        with _tqdm_joblib(tqdm(desc="Factor", total=len(self.partition))):
            self.A = Parallel(n_jobs=n_jobs)(
                delayed(self._svt)(self.selection[part[1]])
                for part in self.partition.items()
            )
        return None

    def combine(self, p: int = 5, rank_oversample: int = 0) -> None:
        # Projection
        rank = []
        for aa in self.A:
            rank.append(aa._svp)

        rank = np.array(rank)
        median_rank = int(np.median(rank))
        print("Median Rank", median_rank)

        G = np.random.randn(
            len(self.selection), median_rank + rank_oversample + p
        )
        Cc = self.block_multiply(G, False)  # M@G
        Dc = self.block_multiply(Cc, True)  # M.T@M@G
        Ec = self.block_multiply(Dc, False)  # M@M.T@M@G
        Fc = self.block_multiply(Ec, True)  # M.T@M@M.T@M@G
        Gc = self.block_multiply(Fc, False)  # M@M.T@M@M.T@M@G
        Q, _, _ = np.linalg.svd(Gc)

        self.Uc = Q[:, : median_rank + rank_oversample]
        # Projection
        self.Mc = np.zeros(
            (self.Uc.shape[1], len(self.selection)), dtype="float32"
        )
        i = 0

        for aa in self.A:
            U = np.array(aa[0])
            S = np.array(aa[1])
            Vt = np.array(aa[2])
            self.Mc[:, self.partition[i]] = (self.Uc.T @ (U * S)) @ Vt

            i += 1

        return None

    def block_multiply(self, B, transpose):  # Parallelize this process
        i = 0
        if transpose:
            C = np.zeros((len(self.selection), B.shape[1]))
        else:
            C = np.zeros((self.A[0][0].shape[0], B.shape[1]))

        for aa in self.A:  # use enumerate here
            if transpose:
                U = np.array(aa[2]).T
                S = np.array(aa[1])
                Vt = np.array(aa[0]).T
                C[self.partition[i], :] = (U * S) @ (Vt @ B)
            else:
                U = np.array(aa[0])
                S = np.array(aa[1])
                Vt = np.array(aa[2])
                C += (U * S) @ (Vt @ B[self.partition[i], :])

            i += 1
        return C

    def _read_in(self, selection):
        # instantiate the reader
        f = self.reader[0]
        iter_var = self.reader.framelist
        out = np.zeros((self.reader.n_mz_bins, len(selection)), dtype=f.dtype)
        C_sp = scis.csr_matrix(
            1
            / (
                self.normalizer[iter_var[selection]]
                / np.median(self.normalizer[iter_var[selection]])
            ),
            dtype="float32",
        )

        for k, i in enumerate(iter_var[selection]):
            f = self.reader[i]._init_csc()
            out[f.indices, k] = f.data

        B = scis.csc_matrix(out, dtype="float32").multiply(
            C_sp
        )  # double check this
        return B

    def _svt(self, selection):
        a = self.overlap[selection]
        b = self._read_in(selection).T  # Read In Data in b
        inst = self.svt(a, b)
        inst.run(k_max=100)
        # obj = [
        #     inst._c[0].todense(),
        #     inst._c[1].todense(),
        #     inst._c[2].todense(),
        #     selection,
        # ]  # Export rank, U, S, Vt and that's it.
        np.savez(
            self.save_path + str(uuid.uuid4()) + ".npz",
            u=inst._c[0].toarray(),
            s=inst._c[1].toarray(),
            vt=inst._c[2].toarray(),
            selection=selection,
        )

        return None

    def _save_intermediate(self, data):
        np.savez(
            self.save_path
            + "/"
            + str(np.random.randint(0, 10000000, 1))
            + ".npz",
            data=data,
        )
        return None

    def combine_from_files(self) -> None:  # recombine from files in here!
        return None


# Got this from internet somewhere
@contextlib.contextmanager
def _tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
