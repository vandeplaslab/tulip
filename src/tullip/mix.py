from typing import Union

import numpy as np
import scipy.sparse as ss
from joblib import Parallel, delayed

import tullip.spatial
import tullip.spectral

# To do:
# - make it possible to link multiple spectra to background
# - fix nar_link and put them on 0


class cynth:
    """
    Generate synthetic cell spatial view:

    Parameters
    ----------

    Attributes
    ----------


    """

    def __init__(
        self,
        spatial: cynth.spatial,
        spectral: cynth.spectral,
    ):
        self._spatial = spatial
        self._spectral = spectral

        # Variables
        self.cell_link = None
        self.nar_link = None
        self.cell_v = None
        self.nar_v = None
        self.overlap = None
        self.overlap_extended = None
        self.scale = None

        pass

    @property
    def cell_spectra(self) -> Union[np.ndarray, None]:
        return (
            (self.cell_v @ self._spectral.cell).astype(np.float32)
            if (self.cell_v is not None)
            else None
        )

    @property
    def nar_spectra(self) -> Union[np.ndarray, None]:
        return (
            (self.nar_v @ self._spectral.nar).astype(np.float32)
            if (self.nar_v is not None)
            else None
        )

    @property
    def spectra(self) -> Union[np.ndarray, None]:
        return (
            np.vstack((self.cell_spectra, self.nar_spectra))
            if (self.cell_spectra is not None)
            else None
        )

    @property
    def mixed(self) -> Union[np.ndarray, None]:
        return self.overlap @ self.spectra if (self.spectra is not None) else None

    @property
    def mixed_noisy(self) -> Union[np.ndarray, None]:
        return self.overlap.multiply(np.random.rand(self.overlap.shape[0], self.overlap.shape[1])) @ self.spectra if (self.spectra is not None) else None    

    @property
    def mixed_block(self) -> Union[np.ndarray, None]:
        return (
            (self.overlap @ self.spectra).reshape(
                (
                    self._spatial.size[0] // self.scale,
                    self._spatial.size[1] // self.scale,
                    -1,
                )
            )
            if (self.spectra is not None)
            else None
        )

    @property
    def class_image(self) -> np.ndarray:
        a = ss.dok_matrix(
            (
                np.prod(self._spatial.size),
                self._spatial.cell_num + self._spatial.nar_num,
            ),
            dtype=np.int32,
        )
        a[
            np.linspace(
                0, np.prod(self._spatial.size) - 1, np.prod(self._spatial.size)
            ),
            self._spatial.view.flatten()-1,
        ] = 1
        b = (a @ np.hstack((self.cell_link, self._spectral.cell_num+self.nar_link))).reshape(
            self._spatial.size
        )
        
        return b

    def link(self) -> None:
        # link spectra to cells following distributions as given in spectral
        for annotation in ["cell", "nar"]:
            self._sample_spectra(
                annotation
            )  # assign a spectral class to each spatial annotation
            self._synthesize(
                annotation
            )  # generate a sparse matrix (annotation_v) that links the spectra to the annotation

        return None

    def downsample(
        self,
        scale: int,
        n_jobs: int = 1,
        dtype: str = "linear",
        sigma: float = 16,
        weights: np.array = np.zeros((1,1)),
    ) -> None:
        self.scale = scale
        getattr(self, "_downsample_" + dtype)(n_jobs, sigma, weights)

        return None

    def _sample_spectra(
        self,
        annotation: str,
    ) -> None:
        # sample distribution (cell and nar)
        s_size = getattr(self._spatial, annotation + "_num")
        size = getattr(self._spectral, annotation + "_num")
        distr = getattr(self._spectral, annotation + "_distribution")
        rand_dist = np.random.choice(
            np.linspace(1, size, size), size=s_size, replace=True, p=distr
        ).astype(np.int32)
        setattr(self, annotation + "_link", rand_dist)  # fill in values

        return None

    def _synthesize(
        self,
        annotation: str,
    ) -> None:
        # create intermediate matrix for synthesizing
        annotation_link = eval("self." + annotation + "_link")
        s_size = getattr(self._spatial, annotation + "_num")
        size = getattr(self._spectral, annotation + "_num")
        a = ss.dok_matrix((s_size, size), dtype=np.int32)
        a[np.linspace(0, s_size - 1, s_size, dtype=np.int32), annotation_link - 1] = 1
        setattr(self, annotation + "_v", a.tocsc())  # fill in values

        return None

    def _downsample_linear(
        self,
        n_jobs: int,
        sigma: float,
        weights,
    ) -> None:
        # make a list of tuples with x_min, y_min
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        n_pix = len(vertices)
        # iterate in parallel over tuples with view
        # a = Parallel(n_jobs=n_jobs)(
        #     delayed(self._bin_overlap)(vertex)
        #     for vertex in vertices
        # )
        a = [self._bin_overlap(vertex) for vertex in vertices]
        self.overlap = ss.hstack(a).T

        return None

    def _bin_overlap(
        self,
        vertex: tuple,
    ):
        a = (
            np.bincount(
                (
                    self._spatial.view[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
                ).flatten(),
                minlength=self._spatial.refactored_nar_ids.size+self._spatial.refactored_cell_ids.size,
            )
            / self.scale**2
        ).astype(np.float32)

        return ss.dok_array(a.reshape(-1,1))

    def extend_overlap(
        self,
        overlap_type: str = "no-nar",
    ) -> None:
        if overlap_type == "no-nar":
            self.overlap_extended = (self.overlap.tocsc()[:,self._spatial.refactored_cell_ids]).tocoo()
            
        return None
      
    def _downsample_gaussian(
        self,
        n_jobs: int,
        sigma: float,
        weights,
    ) -> None:
        # make a list of tuples with x_min, y_min
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        n_pix = len(vertices)
        # iterate in parallel over tuples with view
        # a = Parallel(n_jobs=n_jobs)(
        #     delayed(self._bin_overlap)(vertex)
        #     for vertex in vertices
        # )
        a = [self._bin_overlap_gaussian(vertex, sigma) for vertex in vertices]
        self.overlap = ss.hstack(a).T

        return None
    
    def _downsample_gaussian_weights(
        self,
        n_jobs: int,
        sigma: float,
        weights: np.array,
    ) -> None:
        # make a list of tuples with x_min, y_min
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        n_pix = len(vertices)
        # iterate in parallel over tuples with view
        # a = Parallel(n_jobs=n_jobs)(
        #     delayed(self._bin_overlap)(vertex)
        #     for vertex in vertices
        # )
        a = [self._bin_overlap_gaussian_weights(vertex, sigma, weights) for vertex in vertices]
        self.overlap = ss.hstack(a).T

        return None
    
    def _bin_overlap_gaussian(
        self,
        vertex: tuple,
        sigma: float,
    ):
        
        def gkern(l=5, sig=1.):
            """\
            creates gaussian kernel with side length `l` and a sigma of `sig`
            from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
            """
            ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
            kernel = np.outer(gauss, gauss)
            return kernel / np.sum(kernel)

        a = (
            np.bincount(
                (
                    self._spatial.view[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
                ).flatten(),
                weights=gkern(self.scale, sigma).flatten(),
                minlength=self._spatial.refactored_nar_ids.size+self._spatial.refactored_cell_ids.size,
            )
#            / self.scale**2
        ).astype(np.float32)
        return ss.dok_array(a.reshape(-1,1))

    def extend_overlap(
        self,
        overlap_type: str = "no-nar",
    ) -> None:
        if overlap_type == "no-nar":
            self.overlap_extended = (self.overlap.tocsc()[:,self._spatial.refactored_cell_ids]).tocoo()
            
        return None
    
    def _bin_overlap_gaussian_weights(
        self,
        vertex: tuple,
        sigma: float,
        weights: np.array,
    ):
        
        def gkern(l=5, sig=1.):
            """\
            creates gaussian kernel with side length `l` and a sigma of `sig`
            from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
            """
            ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
            kernel = np.outer(gauss, gauss)
            return kernel / np.sum(kernel)

        cut_weights = weights[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
        
        weights = gkern(self.scale, sigma).flatten()*cut_weights.flatten()
        weights = weights/weights.sum()
        
        a = (
            np.bincount(
                (
                    self._spatial.view[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
                ).flatten(),
                weights=weights,
                minlength=self._spatial.refactored_nar_ids.size+self._spatial.refactored_cell_ids.size,
            )
#            / self.scale**2
        ).astype(np.float32)
        
        return ss.dok_array(a.reshape(-1,1))