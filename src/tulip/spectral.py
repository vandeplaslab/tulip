from typing import Union

import numpy as np
import scipy.sparse as ss
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from sklearn.cluster import KMeans


class SPECTRAL:
    """Generate synthetic cell spectra based on clustering of real spectral data.

    This class implements methods to generate representative spectra for cells and
    non-adherent regions (NAR) using k-means clustering on real spectral data.

    Parameters
    ----------
    cell : np.ndarray or None, optional (default=None)
        Representative cell spectra matrix where each row represents a spectrum.
    nar : np.ndarray or None, optional (default=None)
        Representative non-adherent region spectra matrix.
    cell_distribution : np.ndarray or None, optional (default=None)
        Distribution of cell clusters in the data.
    nar_distribution : np.ndarray or None, optional (default=None)
        Distribution of non-adherent region clusters in the data.

    Attributes
    ----------
    cell : np.ndarray
        Matrix of representative cell spectra.
    nar : np.ndarray
        Matrix of representative non-adherent region spectra.
    cell_distribution : np.ndarray
        Probability distribution of cell clusters.
    nar_distribution : np.ndarray
        Probability distribution of non-adherent region clusters.
    nar_num : int
        Number of NAR spectra.
    cell_num : int
        Number of cell spectra.
    spec_num : int
        Number of spectral points in each spectrum.
    """

    def __init__(
        self,
        cell: Union[np.ndarray, None] = None,
        nar: Union[np.ndarray, None] = None,
        cell_distribution: Union[np.ndarray, None] = None,
        nar_distribution: Union[np.ndarray, None] = None,
    ):
        self._cell = cell
        self._nar = nar
        self._cell_distribution = cell_distribution
        self._nar_distribution = nar_distribution

    @property
    def nar_num(self) -> int:
        """Get the number of NAR spectra.

        Returns
        -------
        int
            Number of NAR spectra, or 0 if no NAR data is present.
        """
        return self._nar.shape[0] if self._nar is not None else 0

    @property
    def cell_num(self) -> int:
        """Get the number of cell spectra.

        Returns
        -------
        int
            Number of cell spectra, or 0 if no cell data is present.
        """
        return self._cell.shape[0] if self._cell is not None else 0

    @property
    def spec_num(self) -> int:
        """Get the number of spectral points in each spectrum.

        Returns
        -------
        int
            Number of spectral points, or 0 if no cell data is present.
        """
        return self._cell.shape[1] if self._cell is not None else 0

    @property
    def cell(self) -> np.ndarray:
        """Get the matrix of representative cell spectra.

        Returns
        -------
        np.ndarray
            Matrix where each row represents a cell spectrum.
        """
        return self._cell

    @cell.setter
    def cell(self, c: np.ndarray) -> None:
        """Set the matrix of representative cell spectra.

        Parameters
        ----------
        c : np.ndarray
            Matrix where each row represents a cell spectrum.
        """
        self._cell = c

    @property
    def nar(self) -> np.ndarray:
        """Get the matrix of representative NAR spectra.

        Returns
        -------
        np.ndarray
            Matrix where each row represents a NAR spectrum.
        """
        return self._nar

    @nar.setter
    def nar(self, n: np.ndarray) -> None:
        """Set the matrix of representative NAR spectra.

        Parameters
        ----------
        n : np.ndarray
            Matrix where each row represents a NAR spectrum.
        """
        self._nar = n

    @property
    def cell_distribution(self) -> np.ndarray:
        """Get the probability distribution of cell clusters.

        Returns
        -------
        np.ndarray
            Array of probabilities for each cell cluster.
        """
        return self._cell_distribution

    @cell_distribution.setter
    def cell_distribution(self, cd: np.ndarray) -> None:
        """Set the probability distribution of cell clusters.

        Parameters
        ----------
        cd : np.ndarray
            Array of probabilities for each cell cluster.
        """
        self._cell_distribution = cd

    @property
    def nar_distribution(self) -> np.ndarray:
        """Get the probability distribution of NAR clusters.

        Returns
        -------
        np.ndarray
            Array of probabilities for each NAR cluster.
        """
        return self._nar_distribution

    @nar_distribution.setter
    def nar_distribution(self, nd: np.ndarray) -> None:
        """Set the probability distribution of NAR clusters.

        Parameters
        ----------
        nd : np.ndarray
            Array of probabilities for each NAR cluster.
        """
        self._nar_distribution = nd

    def from_real(
        self,
        a: Union[ss.dok.dok_matrix, ss.csc.csc_matrix],
        spectra: np.ndarray,
        condition: str,
        n_clus: int,
        annotation: str = "cell",
        kmeans_type: str = "euclidean",
        ind: np.ndarray = None,
        random_state: int = 0,
    ) -> None:
        """Generate representative spectra from real spectral data using k-means clustering.

        Parameters
        ----------
        a : Union[ss.dok.dok_matrix, ss.csc.csc_matrix]
            Sparse matrix containing spectral data.
        spectra : np.ndarray
            Matrix of spectral data where each row is a spectrum.
        condition : str
            String containing the condition to select spectra.
        n_clus : int
            Number of clusters to generate.
        annotation : str, optional (default="cell")
            Type of spectra to generate ("cell" or "nar").
        kmeans_type : str, optional (default="euclidean")
            Distance metric for k-means ("euclidean" or "manhattan").
        ind : np.ndarray, optional (default=None)
            Indices of spectra to use. If None, indices are selected based on condition.
        random_state : int, optional (default=0)
            Random seed for reproducibility.
        """
        ind = ss.find(eval(condition))[0] if (ind is None) else ind
        spectra = self._select_spectra(spectra, ind)

        if kmeans_type == "euclidean":
            centers, distribution = self._apply_kmeans(
                n_clus, random_state, spectra
            )
        elif kmeans_type == "manhattan":
            centers, distribution = self._apply_kmeans_l1(n_clus, spectra)

        setattr(self, annotation, centers)
        setattr(self, annotation + "_distribution", distribution)

    def _apply_kmeans(
        self,
        n_clus: int,
        random_state: int,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply k-means clustering using Euclidean distance.

        Parameters
        ----------
        n_clus : int
            Number of clusters to generate.
        random_state : int
            Random seed for reproducibility.
        spectra : np.ndarray
            Matrix of spectral data where each row is a spectrum.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - Array of cluster centers (representative spectra)
            - Array of cluster probabilities
        """
        kmeans = KMeans(n_clusters=n_clus, random_state=random_state).fit(
            spectra
        )
        return (
            kmeans.cluster_centers_,
            np.bincount(kmeans.labels_) / kmeans.labels_.size,
        )

    def _apply_kmeans_l1(
        self,
        n_clus: int,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply k-means clustering using Manhattan (L1) distance.

        Parameters
        ----------
        n_clus : int
            Number of clusters to generate.
        spectra : np.ndarray
            Matrix of spectral data where each row is a spectrum.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - Array of cluster centers (representative spectra)
            - Array of cluster probabilities
        """
        initial_centers = kmeans_plusplus_initializer(
            spectra, n_clus
        ).initialize()
        manhattan_metric = distance_metric(type_metric.MANHATTAN)
        kmeans_instance = kmeans(
            spectra, initial_centers, metric=manhattan_metric
        )
        kmeans_instance.process()

        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()

        distribution = []
        for clust in clusters:
            distribution.append(len(clust) / len(spectra))

        return (
            np.array(final_centers),
            np.array(distribution),
        )

    def _select_spectra(
        self,
        spectra: np.ndarray,
        ind: Union[np.ndarray, None],
    ) -> np.ndarray:
        """Select spectra based on provided indices.

        Parameters
        ----------
        spectra : np.ndarray
            Matrix of spectral data where each row is a spectrum.
        ind : np.ndarray or None
            Indices of spectra to select.

        Returns
        -------
        np.ndarray
            Selected spectra based on provided indices.
        """
        return spectra[ind, :]
