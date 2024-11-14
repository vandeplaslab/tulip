import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

import tullip.mix
import tullip.spatial
import tullip.spectral


class MEASURES:
    """
    Define measures and plots for assessment of spectral unmixing results

    This class provides various metrics and visualization tools to evaluate
    the quality of spectral unmixing results by comparing recovered spectra
    against ground truth data.

    Parameters
    ----------
    spatial: tullip.spatial
        Object containing spatial information and relationships
    spectral: tullip.spectral
        Object containing spectral data and parameters
    mix: tullip.mix
        Object containing synthetic data and ground truth
    unmix: tullip.unmix
        Object containing unmixing results to be evaluated

    Attributes
    ----------
    _spatial: tullip.spatial
        Stored spatial information
    _spectral: tullip.spectral
        Stored spectral information
    _mix: tullip.mix
        Stored ground truth data
    _unmix: tullip.unmix
        Stored unmixing results
    """

    def __init__(
        self,
        spatial: tullip.spatial,
        spectral: tullip.spectral,
        mix: tullip.mix,
        c: np.array,
    ):
        # Store input objects as class attributes
        self._spatial = spatial
        self._spectral = spectral
        tullip._mix = mix
        self.c = c
        pass

    def nn_entries(
        self,
    ) -> tuple:
        """
        Calculate statistics about non-negative entries in the unmixing results

        Returns
        -------
        tuple:
            b: Array of fraction of non-negative entries per column
            c: Array of fraction of non-negative entries per row
            d: Total fraction of non-negative entries
            e: Sum of all negative entries
        """
        # Calculate fraction of non-negative entries per column
        b = np.sum(np.array(self.c) >= 0, axis=0) / self.c.shape[0]
        # Calculate fraction of non-negative entries per row
        c = np.sum(np.array(self.c) >= 0, axis=1) / self.c.shape[1]
        # Calculate total fraction of non-negative entries
        d = np.sum(np.array(self.c) >= 0) / np.prod(self.c.shape)
        # Calculate sum of all negative entries
        e = self.c[self.c < 0].sum()

        return (b, c, d, e)

    def cell_fit(self) -> tuple:
        """
        Calculate various fit metrics for cell spectra

        Returns
        -------
        tuple:
            a: Array of L2 norm-based fit quality percentages for cells
            b: Array of L2 norm-based fit quality percentages for NAR
            c: Array of maximum relative errors for cells
            d: Array of maximum relative errors for NAR
        """
        # Calculate L2 norm-based fit quality for cells (as percentage)
        a = (
            100
            - np.linalg.norm(
                self._tullip.cell_spectra
                - self.c[self._spatial.refactored_cell_ids, :],
                2,
                axis=1,
            )
            / (np.linalg.norm(self._tullip.cell_spectra, 2, axis=1) + 1e-8)
            * 100
        )

        # Handle NAR (Non-Anatomical Region) spectra differently based on refactored IDs
        if max(self._spatial.refactored_nar_ids) >= self.c.shape[0]:
            nar = np.mean(
                self._tullip.mixed - self._tullip.overlap_extended @ self.c,
                axis=0,
            )
        else:
            nar = self.c[self._spatial.refactored_nar_ids, :]

        # Calculate L2 norm-based fit quality for NAR
        b = (
            100
            - np.linalg.norm(self._tullip.nar_spectra - nar, 2, axis=1)
            / (np.linalg.norm(self._tullip.nar_spectra, 2, axis=1) + 1e-8)
            * 100
        )

        # Calculate maximum relative errors for cells
        c = np.linalg.norm(
            (
                self._tullip.cell_spectra
                - self.c[self._spatial.refactored_cell_ids, :]
            )
            / (self._tullip.cell_spectra + 1e-8),
            np.infty,
            axis=1,
        )

        # Calculate maximum relative errors for NAR
        d = np.linalg.norm(
            (self._tullip.nar_spectra - nar)
            / (self._tullip.nar_spectra + 1e-8),
            np.infty,
            axis=1,
        )

        return (a, b, c, d)

    def fit(self) -> dict:
        """
        Calculate global fit metrics for both cell and NAR spectra

        Returns
        -------
        dict:
            Dictionary containing various fit metrics:
            - Global Cell Fit % (Frobenius): Overall fit quality for cells using Frobenius norm
            - Global NAR Fit % (Frobenius): Overall fit quality for NAR using Frobenius norm
            - Median Cell Fit % (l2): Median of per-cell fit qualities using L2 norm
            - Median NAR Fit % (l2): Median of per-NAR fit qualities using L2 norm
        """
        # Calculate global cell fit using Frobenius norm
        a = (
            100
            - np.linalg.norm(
                self._tullip.cell_spectra
                - self.c[self._spatial.refactored_cell_ids, :],
                "fro",
            )
            / np.linalg.norm(self._tullip.cell_spectra, "fro")
            * 100
        )

        # Handle NAR spectra fitting
        if max(self._spatial.refactored_nar_ids) >= self.c.shape[0]:
            nar = np.mean(
                self._tullip.mixed - self._tullip.overlap_extended @ self.c,
                axis=0,
            )
        else:
            nar = self.c[self._spatial.refactored_nar_ids, :]

        # Calculate global NAR fit using Frobenius norm
        b = (
            100
            - np.linalg.norm(self._tullip.nar_spectra - nar, "fro")
            / np.linalg.norm(self._tullip.nar_spectra, "fro")
            * 100
        )

        # Calculate median of per-cell fit qualities
        c = np.median(
            100
            - np.linalg.norm(
                self._tullip.cell_spectra
                - self.c[self._spatial.refactored_cell_ids, :],
                2,
                axis=1,
            )
            / (np.linalg.norm(self._tullip.cell_spectra, 2, axis=1) + 1e-8)
            * 100
        )

        # Calculate median of per-NAR fit qualities
        d = np.median(
            100
            - np.linalg.norm(self._tullip.nar_spectra - nar, 2, axis=1)
            / (np.linalg.norm(self._tullip.nar_spectra, 2, axis=1) + 1e-8)
            * 100
        )

        return {
            "Global Cell Fit % (Frobenius)": a,
            "Global NAR Fit %(Frobenius)": b,
            "Median Cell Fit % (l2)": c,
            "Median NAR Fit % (l2)": d,
        }

    def panel_plot(
        self,
        entry_list: list,
    ) -> None:
        """
        Create a panel of plots comparing true and recovered spectra

        Parameters
        ----------
        entry_list: list
            List of entries/cells to plot

        Creates a figure with three columns:
        1. True vs recovered spectra
        2. Relative error between true and recovered spectra
        3. Distribution of errors
        """
        # Create subplot grid
        _, axs = plt.subplots(len(entry_list) + 1, 3, figsize=(20, 20))

        # Plot each entry in the list
        for i in range(len(entry_list)):
            # First Column: Plot true and recovered spectra
            axs[i, 0].stem(
                -self._tullip.spectra[entry_list[i], :].T,
                linefmt="white",
                markerfmt=" ",
                label="True Cell #" + str(entry_list[i]),
                basefmt=" ",
            )
            axs[i, 0].stem(
                self.c[entry_list[i], :].T,
                linefmt="C" + str(i),
                markerfmt=" ",
                label="Recovered Cell #" + str(entry_list[i]),
                basefmt=" ",
            )

            # Second Column: Plot relative error
            stm = (
                (
                    self._tullip.spectra[entry_list[i], :].T
                    - self.c[entry_list[i], :].T
                )
                / self._tullip.spectra[entry_list[i], :].T
                * 100
            )
            axs[i, 1].stem(
                stm,
                linefmt="C" + str(i),
                markerfmt=" ",
                label="Cell #" + str(entry_list[i]),
            )

            # Third Column: Plot error distribution
            axs[i, 2].hist(
                self._tullip.spectra[entry_list[i], :]
                - self.c[entry_list[i], :],
                bins=100,
                histtype="step",
                color="C" + str(i),
                linewidth=2,
            )

            # Add formatting
            axs[i, 0].legend()
            axs[i, 2].set_yscale("log")
            axs[i, 0].grid()
            axs[i, 1].grid()
            axs[i, 2].grid()
            axs[i, 0].set_ylabel("Intensity (a.u.)")
            axs[i, 1].set_ylabel("Relative Error (in %)")
            axs[i, 2].set_ylabel("Count")

        # Plot NAR results
        if max(self._spatial.refactored_nar_ids) >= self.c.shape[0]:
            nar = np.mean(
                self._tullip.mixed - self._tullip.overlap_extended @ self.c,
                axis=0,
            )
        else:
            nar = self.c[-1, :].T

        # Add NAR plots in bottom row
        axs[i + 1, 0].stem(
            nar,
            linefmt="C" + str(i + 1),
            markerfmt=" ",
            label="Estimated NAR",
        )
        axs[i + 1, 0].stem(
            -self._spectral.nar[0, :].T,
            linefmt="white",
            markerfmt=" ",
            label="True NAR",
        )

        # Add relative error for NAR
        axs[i + 1, 1].stem(
            (nar - np.squeeze(self._spectral.nar[0, :]).T)
            / np.squeeze(self._spectral.nar[0, :]).T
            * 100,
            linefmt="C" + str(i + 1),
            markerfmt=" ",
            label="NAR",
        )

        # Add error distribution for NAR
        axs[i + 1, 2].hist(
            (nar - np.squeeze(self._spectral.nar[0, :]).T),
            bins=100,
            histtype="step",
            color="C" + str(i + 1),
            linewidth=2,
        )

        # Add final formatting
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

    def bin_plot(
        self,
    ) -> None:
        """
        Create a histogram plot comparing true, estimated, and residual spectra distributions

        Plots three histograms:
        1. Residual (difference between true and estimated)
        2. Ground truth spectra
        3. Estimated spectra
        """
        plt.figure(figsize=(20, 10))

        # Plot residuals
        plt.hist(
            (
                self._tullip.cell_spectra
                - self.c[self._spatial.refactored_cell_ids, :]
            ).flatten(),
            bins=1000,
            histtype="step",
            linewidth=2,
            label="Residual",
        )

        # Plot ground truth
        plt.hist(
            (self._tullip.cell_spectra).flatten(),
            bins=1000,
            histtype="step",
            linewidth=3,
            label="Ground Truth",
        )

        # Plot estimated spectra
        plt.hist(
            (self.c[self._spatial.refactored_cell_ids, :]).flatten(),
            bins=1000,
            histtype="step",
            linewidth=2,
            label="Estimated",
        )

        # Add formatting
        plt.grid()
        plt.xlabel("Intensity")
        plt.ylabel("Entry Count")
        plt.yscale("log")
        plt.xscale("symlog")
        plt.legend()
        plt.show()

    def clustering_measure(
        self,
    ) -> tuple:
        """
        Perform clustering analysis on the unmixing results

        Returns
        -------
        tuple:
            Percentage of correctly matched clusters
            Array of distances between matched cluster centers
        """
        # Apply k-means clustering to unmixed spectra
        a, b = self._apply_kmeans(
            self._spectral.cell_num,
            0,
            self.c[self._spatial.refactored_cell_ids, :],
        )

        # Match means with original clusters using correlation
        coeff = (np.corrcoef(self._spectral.cell, a))[
            : self._spectral.cell_num, self
        ]
        order = np.argmax(coeff, axis=0)

        # compare cluster matching (correct vs. wrong)
        match = 0
        for i in range(self._tullip.cell_link.size):
            match = (
                match + 1
                if (self._tullip.cell_link[i] - 1 == order[b[i]])
                else match + 0
            )

        # check cluster center matching
        dist = (
            100
            * np.linalg.norm(a - self._spectral.cell[order, :], 2, axis=1)
            / np.linalg.norm(self._spectral.cell[order, :], 2, axis=1)
        )

        return ((100 * match / self._tullip.cell_link.size), dist)

    def apply_kmeans(
        self,
        n_clus: int,
        random_state: int,
        spectra: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply K-means clustering to spectral data.

        Performs K-means clustering on the input spectral data using scikit-learn's
        implementation. Returns both cluster centers and labels for each data point.

        Parameters
        ----------
        n_clus : int
            Number of clusters to form
        random_state : int
            Random seed for reproducibility
        spectra : np.ndarray
            Input spectral data array to cluster, shape (n_samples, n_features)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            First array contains cluster centers, shape (n_clusters, n_features)
            Second array contains cluster labels, shape (n_samples,)
        """
        kmeans = KMeans(n_clusters=n_clus, random_state=random_state).fit(
            spectra
        )
        return (kmeans.cluster_centers_, kmeans.labels_)

    def fit_view(self, fit_array: np.ndarray) -> np.ndarray:
        """
        Map fitted values to spatial view coordinates.

        Creates a new array matching the spatial view dimensions and maps values
        from fit_array to their corresponding positions based on the spatial view index.
        Non-mapped positions are set to NaN.

        Parameters
        ----------
        fit_array : np.ndarray
            Array of values to map to spatial positions

        Returns
        -------
        np.ndarray
            Array with same shape as spatial view containing mapped values,
            with NaN for unmapped positions
        """
        a = np.zeros_like(self._spatial.view).astype(np.float32)
        a[a == 0] = np.nan
        for idx, spec in enumerate(fit_array):
            a[np.where(self._spatial.view == idx)] = spec
        return a

    def fit_category(self) -> tuple[float, float, float]:
        """
        Calculate fit quality metrics for different overlap categories.

        Computes Frobenius norm-based similarity metrics between cell spectra and
        unmixed components for three categories of overlap:
        1. Cells with overlap sum < 1
        2. Cells with overlap sum between 1 and 2
        3. Cells with overlap sum >= 2 (excluding refactored NAR IDs)

        Returns
        -------
        tuple[float, float, float]
            Three similarity scores (as percentages) corresponding to each category:
            - First value: Score for cells with overlap < 1
            - Second value: Score for cells with overlap between 1 and 2
            - Third value: Score for cells with overlap >= 2

        Notes
        -----
        Higher percentage values indicate better fit between cell spectra and
        unmixed components.
        """
        q1 = np.where(self._tullip.overlap.sum(axis=0) < 1)[1]
        q2 = np.where(
            np.logical_and(
                self._tullip.overlap.sum(axis=0) >= 1,
                self._tullip.overlap.sum(axis=0) < 2,
            )
        )[1]
        q3 = np.setdiff1d(
            np.where(self._tullip.overlap.sum(axis=0) >= 2)[1],
            self._spatial.refactored_nar_ids,
        )

        print(q1.shape, q2.shape, q3.shape)
        print(
            (
                self._tullip.cell_spectra[q1, :]
                - self.c[self._spatial.refactored_cell_ids[q1], :]
            ).shape
        )

        a = (
            100
            - np.linalg.norm(
                self._tullip.cell_spectra[q1, :]
                - self.c[self._spatial.refactored_cell_ids[q1], :],
                "fro",
            )
            / np.linalg.norm(self._tullip.cell_spectra[q1, :], "fro")
            * 100
        )

        b = (
            100
            - np.linalg.norm(
                self._tullip.cell_spectra[q2, :]
                - self.c[self._spatial.refactored_cell_ids[q2], :],
                "fro",
            )
            / np.linalg.norm(self._tullip.cell_spectra[q2, :], "fro")
            * 100
        )

        c = (
            100
            - np.linalg.norm(
                self._tullip.cell_spectra[q3, :]
                - self.c[self._spatial.refactored_cell_ids[q3], :],
                "fro",
            )
            / np.linalg.norm(self._tullip.cell_spectra[q3, :], "fro")
            * 100
        )
        return (a, b, c)
