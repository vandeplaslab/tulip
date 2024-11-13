import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import tullip.cynth
import tullip.spatial
import tullip.spectral
import tullip.unmix


class measures:
    """
    Define measures and plots for assessment

    Parameters
    ----------

    Attributes
    ----------


    """

    def __init__(
        self,
        spatial: cynth.spatial,
        spectral: cynth.spectral,
        cynth: cynth.cynth,
        unmix: cynth.unmix,
    ):
        self._spatial = spatial
        self._spectral = spectral
        self._cynth = cynth
        self._unmix = unmix
        pass

    def nn_entries(
        self,
    ) -> tuple:
        # count relative number of non-negative entries, per row, column and total
        b = np.sum(np.array(self._unmix.c) >= 0, axis=0) / self._unmix.c.shape[0]
        c = np.sum(np.array(self._unmix.c) >= 0, axis=1) / self._unmix.c.shape[1]
        d = np.sum(np.array(self._unmix.c) >= 0) / np.prod(self._unmix.c.shape)
        e = self._unmix.c[self._unmix.c < 0].sum()

        return (b, c, d, e)
    
    def cell_fit(self) -> tuple:
        a = (
            100
            - np.linalg.norm(self._cynth.cell_spectra - self._unmix.c[self._spatial.refactored_cell_ids,:], 2, axis=1)
            / (np.linalg.norm(self._cynth.cell_spectra, 2, axis=1)+1e-8)
            * 100
        )
        if max(self._spatial.refactored_nar_ids) >= self._unmix.c.shape[0]:
            nar = np.mean(self._cynth.mixed-self._cynth.overlap_extended@self._unmix.c, axis = 0)
        else:
            nar = self._unmix.c[self._spatial.refactored_nar_ids,:]     
        b = (
            100
            - np.linalg.norm(self._cynth.nar_spectra - nar, 2, axis=1)
            / (np.linalg.norm(self._cynth.nar_spectra, 2, axis=1)+1e-8)
            * 100
        )
        c = np.linalg.norm(
            (self._cynth.cell_spectra - self._unmix.c[self._spatial.refactored_cell_ids,:]) / (self._cynth.cell_spectra+1e-8),
            np.infty,
            axis=1,
        )
        d = np.linalg.norm(
            (self._cynth.nar_spectra - nar) / (self._cynth.nar_spectra+1e-8),
            np.infty,
            axis=1,
        ) 

        return (a, b, c, d)

    def fit(self) -> dict:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        # a = (
        #     100
        #     - np.linalg.norm(self._cynth.spectra - self._unmix.c, "fro")
        #     / np.linalg.norm(self._cynth.spectra, "fro")
        #     * 100
        # )
        # b = (
        #     100
        #     - np.linalg.norm(self._cynth.spectra - self._unmix.c, 2)
        #     / np.linalg.norm(self._cynth.spectra, 2)
        #     * 100
        # )
        a = (
            100
            - np.linalg.norm(self._cynth.cell_spectra - self._unmix.c[self._spatial.refactored_cell_ids,:], "fro")
            / np.linalg.norm(self._cynth.cell_spectra, "fro")
            * 100
        )
        if max(self._spatial.refactored_nar_ids) >= self._unmix.c.shape[0]:
            nar = np.mean(self._cynth.mixed-self._cynth.overlap_extended@self._unmix.c, axis = 0)
        else:
            nar = self._unmix.c[self._spatial.refactored_nar_ids,:]           
        b = (
            100
            - np.linalg.norm(self._cynth.nar_spectra - nar, "fro")
            / np.linalg.norm(self._cynth.nar_spectra, "fro")
            * 100
        )
        
        c = np.median(
            100
            - np.linalg.norm(self._cynth.cell_spectra - self._unmix.c[self._spatial.refactored_cell_ids,:], 2, axis=1)
            / (np.linalg.norm(self._cynth.cell_spectra, 2, axis=1)+1e-8)
            * 100
        )
        d = np.median(
            100
            - np.linalg.norm(self._cynth.nar_spectra - nar, 2, axis=1)
            / (np.linalg.norm(self._cynth.nar_spectra, 2, axis=1)+1e-8)
            * 100
        )
        
        return {'Global Cell Fit % (Frobenius)' : a, 
                'Global NAR Fit %(Frobenius)' : b,
                'Median Cell Fit % (l2)' : c,
                'Median NAR Fit % (l2)' : d}

    def panel_plot(
        self,
        entry_list: list,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        _, axs = plt.subplots(len(entry_list) + 1, 3, figsize=(20, 20))
        for i in range(len(entry_list)):
            # First Column
            axs[i, 0].stem(
                -self._cynth.spectra[entry_list[i], :].T,
                linefmt="white",
                markerfmt=" ",
                label="True Cell #" + str(entry_list[i]),
                basefmt=" ",
            )
            axs[i, 0].stem(
                self._unmix.c[entry_list[i], :].T,
                linefmt="C" + str(i),
                markerfmt=" ",
                label="Recovered Cell #" + str(entry_list[i]),
                basefmt=" ",
            )
            # Second Column
            stm = (
                (
                    self._cynth.spectra[entry_list[i], :].T
                    - self._unmix.c[entry_list[i], :].T
                )
                / self._cynth.spectra[entry_list[i], :].T
                * 100
            )
            axs[i, 1].stem(
                stm,
                linefmt="C" + str(i),
                markerfmt=" ",
                label="Cell #" + str(entry_list[i]),
            )
            # Third Column
            axs[i, 2].hist(
                self._cynth.spectra[entry_list[i], :] - self._unmix.c[entry_list[i], :],
                bins=100,
                histtype="step",
                color="C" + str(i),
                linewidth=2,
            )

            axs[i, 0].legend()

            axs[i, 2].set_yscale("log")

            axs[i, 0].grid()
            axs[i, 1].grid()
            axs[i, 2].grid()

            axs[i, 0].set_ylabel("Intensity (a.u.)")
            axs[i, 1].set_ylabel("Relative Error (in %)")
            axs[i, 2].set_ylabel("Count")

        # Plot NAR: rewrite this
        if max(self._spatial.refactored_nar_ids) >= self._unmix.c.shape[0]:
            nar = np.mean(self._cynth.mixed-self._cynth.overlap_extended@self._unmix.c, axis = 0)
        else:
            nar = self._unmix.c[-1, :].T
        
        
        axs[i + 1, 0].stem(
            nar,
            linefmt="C" + str(i + 1),
            markerfmt=" ",
            label="Estimated NAR",
        )
        axs[i + 1, 0].stem(
            -self._spectral.nar[0,:].T, linefmt="white", markerfmt=" ", label="True NAR"
        )
        axs[i + 1, 1].stem(
            (nar - np.squeeze(self._spectral.nar[0,:]).T)
            / np.squeeze(self._spectral.nar[0,:]).T
            * 100,
            linefmt="C" + str(i + 1),
            markerfmt=" ",
            label="NAR",
        )
        axs[i + 1, 2].hist(
            (nar - np.squeeze(self._spectral.nar[0,:]).T),
            bins=100,
            histtype="step",
            color="C" + str(i + 1),
            linewidth=2,
        )

        axs[i + 1, 0].grid()
        axs[i + 1, 0].set_ylabel("Intensity (a.u.)")
        axs[i + 1, 0].set_xlabel("m/z bin")
        axs[i + 1, 0].legend()

        axs[i + 1, 1].set_ylabel("Relative Error (in %)")
        axs[i + 1, 1].grid()
        axs[i + 1, 1].legend()
        axs[i + 1, 1].set_xlabel("m/z bin")

        axs[i + 1, 2].set_ylabel("Count")
        axs[i + 1, 2].set_xlabel("Residual Intensity")
        axs[i + 1, 2].set_yscale("log")
        axs[i + 1, 2].grid()

        axs[0, 0].set_title("Spectrum")
        axs[0, 1].set_title("Spectral Relative Error")
        axs[0, 2].set_title("Spectral Error Distribution")

        plt.show()

        return None

    def bin_plot(
        self,
    ) -> None:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        plt.figure(figsize=(20, 10))
        plt.hist(
            (
                self._cynth.cell_spectra
                - self._unmix.c[self._spatial.refactored_cell_ids, :]
            ).flatten(),
            bins=1000,
            histtype="step",
            linewidth=2,
            label="Residual",
        )
        plt.hist(
            (self._cynth.cell_spectra).flatten(),
            bins=1000,
            histtype="step",
            linewidth=3,
            label="Ground Truth",
        )
        plt.hist(
            (self._unmix.c[self._spatial.refactored_cell_ids, :]).flatten(),
            bins=1000,
            histtype="step",
            linewidth=2,
            label="Estimated",
        )
        plt.grid()
        plt.xlabel("Intensity")
        plt.ylabel("Entry Count")
        plt.yscale("log")
        plt.xscale("symlog")
        plt.legend()
        plt.show()

        return None

    def clustering_measure(
        self,
    ) -> tuple:
        """

        Parameters
        ----------
        a :

        Returns
        -------
        a :

        """
        a, b = self._apply_kmeans(
            self._spectral.cell_num,
            0,
            self._unmix.c[self._spatial.refactored_cell_ids, :],
        )

        # match means with original clusters
        coeff = (np.corrcoef(self._spectral.cell, a))[:self._spectral.cell_num, self._spectral.cell_num:]
        order = np.argmax(coeff, axis=0)

        # compare cluster matching (correct vs. wrong)
        match = 0
        for i in range(self._cynth.cell_link.size):
            match = (
                match + 1
                if (self._cynth.cell_link[i] - 1 == order[b[i]])
                else match + 0
            )

        # check cluster center matching
        dist = (
            100
            * np.linalg.norm(a - self._spectral.cell[order, :], 2, axis=1)
            / np.linalg.norm(self._spectral.cell[order, :], 2, axis=1)
        )

        return ((100 * match / self._cynth.cell_link.size), dist)

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

        return (kmeans.cluster_centers_, kmeans.labels_)
    
    def fit_view(
        self,
        fit_array :  np.ndarray) -> np.ndarray:
        
        a = np.zeros_like(self._spatial.view).astype(np.float32)
        a[a == 0] = np.nan
        for idx, spec in enumerate(fit_array):
            a[np.where(self._spatial.view == idx)] = spec
    
        return a

    
    def fit_category(self):
        q1 = np.where(self._cynth.overlap.sum(axis=0) < 1)[1], 
        q2 = np.where(np.logical_and(self._cynth.overlap.sum(axis=0) >=1, self._cynth.overlap.sum(axis=0) < 2))[1]
        q3 = np.setdiff1d(np.where(self._cynth.overlap.sum(axis=0) >= 2)[1], self._spatial.refactored_nar_ids)
        print(q1.shape, q2.shape, q3.shape)
        print((self._cynth.cell_spectra[q1,:] - self._unmix.c[self._spatial.refactored_cell_ids[q1],:]).shape)
        a = (
            100
            - np.linalg.norm(self._cynth.cell_spectra[q1,:] - self._unmix.c[self._spatial.refactored_cell_ids[q1],:], "fro")
            / np.linalg.norm(self._cynth.cell_spectra[q1,:] , "fro")
            * 100
        )
        
        b = (
            100
            - np.linalg.norm(self._cynth.cell_spectra[q2,:] - self._unmix.c[self._spatial.refactored_cell_ids[q2],:], "fro")
            / np.linalg.norm(self._cynth.cell_spectra[q2,:] , "fro")
            * 100
        )
            
        c = (
            100
            - np.linalg.norm(self._cynth.cell_spectra[q3,:] - self._unmix.c[self._spatial.refactored_cell_ids[q3],:], "fro")
            / np.linalg.norm(self._cynth.cell_spectra[q3,:] , "fro")
            * 100
        )
        return (a, b, c)