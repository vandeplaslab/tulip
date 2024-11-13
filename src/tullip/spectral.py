from typing import Union

import numpy as np
import scipy.sparse as ss
from sklearn.cluster import KMeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric

# to do :
# - finish comments


class spectral:
    """Generate synthetic cell spectra:

    Parameters
    ----------

    Attributes
    ----------


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

        pass

    @property
    def nar_num(self) -> int:
        return self._nar.shape[0] if self._nar is not None else 0

    @property
    def cell_num(self) -> int:
        return self._cell.shape[0] if self._cell is not None else 0

    @property
    def spec_num(self) -> int:
        return self._cell.shape[1] if self._cell is not None else 0

    @property
    def cell(self) -> np.ndarray:
        return self._cell

    @cell.setter
    def cell(
        self,
        c: np.ndarray,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        self._cell = c

        return None

    @property
    def nar(self) -> np.ndarray:
        return self._nar

    @nar.setter
    def nar(
        self,
        n: np.ndarray,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        self._nar = n

        return None

    @property
    def cell_distribution(self) -> np.ndarray:
        return self._cell_distribution

    @cell_distribution.setter
    def cell_distribution(
        self,
        cd: np.ndarray,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        self._cell_distribution = cd

        return None

    @property
    def nar_distribution(self) -> np.ndarray:
        return self._nar_distribution

    @nar_distribution.setter
    def nar_distribution(
        self,
        nd: np.ndarray,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        self._nar_distribution = nd

        return None

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
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        ind = ss.find(eval(condition))[0] if (ind is None) else ind  # select indices
        spectra = self._select_spectra(
            spectra, ind
        )  # select spectra matching to indices
        if kmeans_type == "euclidean":
            centers, distribution = self._apply_kmeans(
                n_clus, random_state, spectra
            )  # apply kmeans
        elif kmeans_type == "manhattan":
            centers, distribution = self._apply_kmeans_l1(
                n_clus, spectra
            )  # apply kmeans            
        setattr(self, annotation, centers)  # fill in values
        setattr(self, annotation + "_distribution", distribution)  # fill in values

        return None

    def _apply_kmeans(
        self,
        n_clus: int,
        random_state: int,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply kmeans clustering

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        kmeans = KMeans(n_clusters=n_clus, random_state=random_state).fit(spectra)

        return (
            kmeans.cluster_centers_,
            np.bincount(kmeans.labels_) / kmeans.labels_.size,
        )
    
    
    def _apply_kmeans_l1(
        self,
        n_clus: int,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply l1-kmeans clustering

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
         # Prepare initial centers using K-Means++ method.
        initial_centers = kmeans_plusplus_initializer(spectra, n_clus).initialize()
        manhattan_metric = distance_metric(type_metric.MANHATTAN)

        # create instance of K-Means using specific distance metric:
        kmeans_instance = kmeans(spectra, initial_centers, metric=manhattan_metric)

        # Run cluster analysis and obtain results.
        kmeans_instance.process()
        clusters = kmeans_instance.get_clusters()
        final_centers = kmeans_instance.get_centers()
        
        distribution = []
        for clust in clusters:
            distribution.append(len(clust)/len(spectra))

        return (
            np.array(final_centers),
            np.array(distribution),
        )

    def _select_spectra(
        self,
        spectra: np.ndarray,
        ind: Union[np.ndarray, None],
    ) -> np.ndarray:
        """
        Select spectra matching to indices

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        return spectra[ind, :]
