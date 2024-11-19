from typing import Union

import numpy as np
import scipy.sparse as ss

import tulip.spatial
import tulip.spectral


class MIX:
    """
    Generate synthetic cell spatial views by mixing spatial and spectral data.
    This class combines cellular spatial information with spectral signatures
    to create synthetic cell visualization data.

    Parameters
    ----------
    spatial : tulip.spatial
        Object containing spatial information about cells and non-annotated regions (nar)
    spectral : tulip.spectral
        Object containing spectral signatures for cells and non-annotated regions

    Attributes
    ----------
    cell_link : array-like
        Links between spatial cell annotations and spectral cell classes
    nar_link : array-like
        Links between spatial non-annotated regions and spectral nar classes
    cell_v : sparse matrix
        Matrix linking cells to their spectral signatures
    nar_v : sparse matrix
        Matrix linking non-annotated regions to their spectral signatures
    overlap : sparse matrix
        Matrix representing spatial overlap between cells in downsampled view
    overlap_extended : sparse matrix
        Extended version of overlap matrix (typically excluding nar)
    scale : int
        Downsampling scale factor
    """

    def __init__(
        self,
        spatial: tulip.spatial,
        spectral: tulip.spectral,
    ):
        self._spatial = spatial
        self._spectral = spectral

        # Initialize instance variables
        self.cell_link = None  # Links spatial cells to spectral classes
        self.nar_link = None  # Links spatial nar to spectral classes
        self.cell_v = None  # Cell-spectra linking matrix
        self.nar_v = None  # NAR-spectra linking matrix
        self.overlap = None  # Spatial overlap matrix
        self.overlap_extended = None  # Extended overlap matrix
        self.scale = None  # Downsampling scale factor

    @property
    def cell_spectra(self) -> Union[np.ndarray, None]:
        """
        Calculate combined cell spectral signatures.

        Returns
        -------
        Union[np.ndarray, None]
            A numpy array containing the combined cell spectral signatures if cell_v exists,
            otherwise None. The array is computed as the matrix product of cell_v and
            _spectral.cell, cast to float32 type.
        """
        return (
            (self.cell_v @ self._spectral.cell).astype(np.float32)
            if (self.cell_v is not None)
            else None
        )

    @property
    def nar_spectra(self) -> Union[np.ndarray, None]:
        """
        Calculate combined non-annotated region spectral signatures.

        Returns
        -------
        Union[np.ndarray, None]
            A numpy array containing the combined non-annotated region spectral signatures
            if nar_v exists, otherwise None. The array is computed as the matrix product
            of nar_v and _spectral.nar, cast to float32 type.
        """
        return (
            (self.nar_v @ self._spectral.nar).astype(np.float32)
            if (self.nar_v is not None)
            else None
        )

    @property
    def spectra(self) -> Union[np.ndarray, None]:
        """
        Combine cell and non-annotated region spectra into a single array.

        Returns
        -------
        Union[np.ndarray, None]
            A vertically stacked numpy array containing both cell and non-annotated region
            spectral signatures if cell_spectra exists, otherwise None. The array is created
            by vertically stacking cell_spectra and nar_spectra.
        """
        return (
            np.vstack((self.cell_spectra, self.nar_spectra))
            if (self.cell_spectra is not None)
            else None
        )

    @property
    def mixed(self) -> Union[np.ndarray, None]:
        """
        Generate mixed spectral data using overlap matrix.

        Returns
        -------
        Union[np.ndarray, None]
            A numpy array containing the mixed spectral data if spectra exists, otherwise None.
            The array is computed as the matrix product of the overlap matrix and spectra.
        """
        return (
            self.overlap @ self.spectra if (self.spectra is not None) else None
        )

    @property
    def mixed_noisy(self) -> Union[np.ndarray, None]:
        """
        Generate mixed spectral data with random noise applied to overlap matrix.

        Returns
        -------
        Union[np.ndarray, None]
            A numpy array containing the noisy mixed spectral data if spectra exists,
            otherwise None. The array is computed by multiplying the overlap matrix with
            random values and then performing matrix multiplication with spectra.
        """
        return (
            self.overlap.multiply(
                np.random.rand(self.overlap.shape[0], self.overlap.shape[1])
            )
            @ self.spectra
            if (self.spectra is not None)
            else None
        )

    @property
    def mixed_block(self) -> Union[np.ndarray, None]:
        """
        Reshape mixed spectral data into blocks based on scale.

        Returns
        -------
        Union[np.ndarray, None]
            A 3D numpy array containing the reshaped mixed spectral data if spectra exists,
            otherwise None. The array is reshaped according to spatial dimensions divided
            by the scale factor.
        """
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
        """
        Generate image showing class assignments for each spatial position.

        Returns
        -------
        np.ndarray
            A 2D numpy array representing the class assignments across spatial positions.
            The array is created by:
            1. Creating a sparse matrix for class assignments
            2. Assigning classes based on spatial view
            3. Combining cell and non-annotated region links
            4. Reshaping to match spatial dimensions
        """
        # Create sparse matrix for class assignments
        a = ss.dok_matrix(
            (
                np.prod(self._spatial.size),
                self._spatial.cell_num + self._spatial.nar_num,
            ),
            dtype=np.int32,
        )
        # Assign classes based on spatial view
        a[
            np.linspace(
                0, np.prod(self._spatial.size) - 1, np.prod(self._spatial.size)
            ),
            self._spatial.view.flatten() - 1,
        ] = 1
        # Combine cell and nar links and reshape to spatial dimensions
        b = (
            a
            @ np.hstack(
                (self.cell_link, self._spectral.cell_num + self.nar_link)
            )
        ).reshape(self._spatial.size)
        return b

    def link(self) -> None:
        """
        Create links between spatial annotations and spectral classes.

        This method processes both cell and non-annotated region (nar) annotations by:
        1. Assigning spectral classes through _sample_spectra()
        2. Generating linking matrices through _synthesize()

        Returns
        -------
        None
            This method modifies the object's state but does not return any value.
        """
        # Process both cell and non-annotated region annotations
        for annotation in ["cell", "nar"]:
            self._sample_spectra(annotation)  # Assign spectral classes
            self._synthesize(annotation)  # Generate linking matrices
        return None

    def downsample(
        self,
        scale: int,
        n_jobs: int = 1,
        dtype: str = "linear",
        sigma: float = 16,
        weights: np.array = np.zeros((1, 1)),
    ) -> None:
        """
        Downsample the spatial view using the specified method.

        This method reduces the resolution of the spatial view by a given scale factor
        using one of several available downsampling algorithms. The downsampling can be
        performed using either linear averaging or Gaussian smoothing, with optional
        position-specific weights.

        Parameters
        ----------
        scale : int
            Downsampling factor. The output dimensions will be original_size/scale.
            Must be a positive integer.

        n_jobs : int, optional
            Number of parallel jobs to run for computation. Currently unused.
            Default is 1.

        dtype : str, optional
            Downsampling method to use. Must be one of:
            - 'linear': Simple averaging within each block
            - 'gaussian': Gaussian-weighted averaging
            Default is 'linear'.

        sigma : float, optional
            Standard deviation for Gaussian kernel when using gaussian downsampling.
            Larger values result in more smoothing.
            Only used when dtype='gaussian'.
            Default is 16.

        weights : np.array, optional
            Array of position-specific weights to apply during downsampling.
            Must match the spatial dimensions of the input if provided.
            Only used for weighted gaussian downsampling.
            Default is np.zeros((1, 1)).

        Returns
        -------
        None
            Updates the object's state by:
            1. Setting the scale attribute
            2. Computing and storing the downsampled overlap matrix

        Notes
        -----
        The actual downsampling computation is delegated to one of three methods:
        - _downsample_linear
        - _downsample_gaussian
        - _downsample_gaussian_weights

        The choice is determined by the dtype parameter and presence of weights.
        """
        self.scale = scale
        getattr(self, "_downsample_" + dtype)(n_jobs, sigma, weights)

    def _sample_spectra(
        self,
        annotation: str,
    ) -> None:
        """
        Sample spectral classes for spatial annotations following given distributions.

        This method assigns spectral classes to spatial annotations by sampling from a
        specified probability distribution. The sampling process uses the dimensions and
        distributions defined in the spatial and spectral properties of the object.

        Parameters
        ----------
        annotation : str
            Type of annotation to process, must be either 'cell' or 'nar' (non-annotated region)

        Returns
        -------
        None
            Updates the object's state by setting the '{annotation}_link' attribute
        """
        # Get dimensions and distribution for sampling
        s_size = getattr(self._spatial, annotation + "_num")
        size = getattr(self._spectral, annotation + "_num")
        distr = getattr(self._spectral, annotation + "_distribution")

        # Sample from distribution
        rand_dist = np.random.choice(
            np.linspace(1, size, size), size=s_size, replace=True, p=distr
        ).astype(np.int32)
        setattr(self, annotation + "_link", rand_dist)

    def _synthesize(
        self,
        annotation: str,
    ) -> None:
        """
        Create sparse matrix linking spatial annotations to spectral classes.

        This method generates a sparse matrix that represents the connections between
        spatial annotations and their corresponding spectral classes. The matrix is
        created in DOK format and converted to CSC format for efficient operations.

        Parameters
        ----------
        annotation : str
            Type of annotation to process, must be either 'cell' or 'nar' (non-annotated region)

        Returns
        -------
        None
            Updates the object's state by setting the '{annotation}_v' attribute
        """
        annotation_link = eval("self." + annotation + "_link")
        s_size = getattr(self._spatial, annotation + "_num")
        size = getattr(self._spectral, annotation + "_num")

        # Create sparse matrix for links
        a = ss.dok_matrix((s_size, size), dtype=np.int32)
        a[
            np.linspace(0, s_size - 1, s_size, dtype=np.int32),
            annotation_link - 1,
        ] = 1
        setattr(self, annotation + "_v", a.tocsc())

    def _downsample_linear(
        self,
        n_jobs: int,
        sigma: float,
        weights,
    ) -> None:
        """
        Perform linear downsampling of spatial view.

        This method creates a downsampled version of the spatial view by generating
        vertices for downsampled blocks and calculating their overlap values.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run
        sigma : float
            Smoothing parameter (not used in linear downsampling)
        weights
            Weights for downsampling (not used in linear downsampling)

        Returns
        -------
        None
            Updates the object's overlap attribute
        """
        # Generate vertex coordinates for downsampled blocks
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        # Calculate overlap for each vertex
        a = [self._bin_overlap(vertex) for vertex in vertices]
        self.overlap = ss.hstack(a).T

    def _bin_overlap(
        self,
        vertex: tuple,
    ) -> ss.dok_array:
        """
        Calculate overlap values for a specific vertex in downsampled view.

        This method computes the normalized count of each class within a block
        defined by the vertex coordinates.

        Parameters
        ----------
        vertex : tuple
            (x, y) coordinates of vertex in the downsampled grid

        Returns
        -------
        ss.dok_array
            Sparse array containing normalized class counts for the block
        """
        # Count occurrences of each class in the block and normalize
        a = (
            np.bincount(
                (
                    self._spatial.view[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
                ).flatten(),
                minlength=self._spatial.refactored_nar_ids.size
                + self._spatial.refactored_cell_ids.size,
            )
            / self.scale**2
        ).astype(np.float32)

        return ss.dok_array(a.reshape(-1, 1))

    def extend_overlap(
        self,
        overlap_type: str = "no-nar",
    ) -> None:
        """
        Extend overlap matrix based on specified type.

        This method modifies the overlap matrix according to the specified extension type.
        Currently only supports 'no-nar' type which excludes non-annotated regions.

        Parameters
        ----------
        overlap_type : str, optional
            Type of extension to perform, default is 'no-nar'

        Returns
        -------
        None
            Updates the object's overlap_extended attribute
        """
        if overlap_type == "no-nar":
            self.overlap_extended = (
                self.overlap.tocsc()[:, self._spatial.refactored_cell_ids]
            ).tocoo()

    def _downsample_gaussian(
        self,
        n_jobs: int,
        sigma: float,
        weights,
    ) -> None:
        """
        Perform Gaussian downsampling of spatial view.

        This method creates a downsampled version of the spatial view using
        Gaussian weighting for smoothing.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run
        sigma : float
            Standard deviation for Gaussian kernel
        weights
            Weights for downsampling (not used in basic Gaussian downsampling)

        Returns
        -------
        None
            Updates the object's overlap attribute
        """
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        a = [self._bin_overlap_gaussian(vertex, sigma) for vertex in vertices]
        self.overlap = ss.hstack(a).T

    def _downsample_gaussian_weights(
        self,
        n_jobs: int,
        sigma: float,
        weights: np.array,
    ) -> None:
        """
        Perform weighted Gaussian downsampling of spatial view.

        This method creates a downsampled version of the spatial view using both
        Gaussian smoothing and additional weights for each position.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run
        sigma : float
            Standard deviation for Gaussian kernel
        weights : np.array
            Array of weights to apply during downsampling

        Returns
        -------
        None
            Updates the object's overlap attribute
        """
        vertices = [
            (a, b)
            for a in range(self._spatial.size[0] // self.scale)
            for b in range(self._spatial.size[1] // self.scale)
        ]
        a = [
            self._bin_overlap_gaussian_weights(vertex, sigma, weights)
            for vertex in vertices
        ]
        self.overlap = ss.hstack(a).T

    def _bin_overlap_gaussian(
        self,
        vertex: tuple,
        sigma: float,
    ) -> ss.dok_array:
        """
        Calculate Gaussian-weighted overlap values for a vertex.

        This method computes the class distribution within a block using
        Gaussian-weighted averaging.

        Parameters
        ----------
        vertex : tuple
            (x, y) coordinates of vertex in the downsampled grid
        sigma : float
            Standard deviation for Gaussian kernel

        Returns
        -------
        ss.dok_array
            Sparse array containing Gaussian-weighted class distributions
        """

        def gkern(length=5, sig=1.0):
            """
            Create 2D Gaussian kernel.

            Parameters
            ----------
            length : int, optional
                Size of the kernel, default is 5
            sig : float, optional
                Standard deviation of Gaussian, default is 1.0

            Returns
            -------
            np.ndarray
                2D normalized Gaussian kernel
            """
            ax = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
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
                minlength=self._spatial.refactored_nar_ids.size
                + self._spatial.refactored_cell_ids.size,
            )
        ).astype(np.float32)
        return ss.dok_array(a.reshape(-1, 1))

    def _bin_overlap_gaussian_weights(
        self,
        vertex: tuple,
        sigma: float,
        weights: np.array,
    ) -> ss.dok_array:
        """
        Calculate weighted Gaussian overlap values for a vertex.

        This method computes the class distribution within a block using both
        Gaussian smoothing and additional position-specific weights.

        Parameters
        ----------
        vertex : tuple
            (x, y) coordinates of vertex in the downsampled grid
        sigma : float
            Standard deviation for Gaussian kernel
        weights : np.array
            Array of weights to apply to each position

        Returns
        -------
        ss.dok_array
            Sparse array containing weighted Gaussian class distributions
        """

        def gkern(length=5, sig=1.0):
            """
            Create 2D Gaussian kernel.

            Parameters
            ----------
            length : int, optional
                Size of the kernel, default is 5
            sig : float, optional
                Standard deviation of Gaussian, default is 1.0

            Returns
            -------
            np.ndarray
                2D normalized Gaussian kernel
            """
            ax = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
            gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
            kernel = np.outer(gauss, gauss)
            return kernel / np.sum(kernel)

        # Extract weights for current block
        cut_weights = weights[
            vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
            vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
        ]

        # Combine Gaussian kernel with weights
        weights = gkern(self.scale, sigma).flatten() * cut_weights.flatten()
        weights = weights / weights.sum()

        a = (
            np.bincount(
                (
                    self._spatial.view[
                        vertex[0] * self.scale : (vertex[0] + 1) * self.scale,
                        vertex[1] * self.scale : (vertex[1] + 1) * self.scale,
                    ]
                ).flatten(),
                weights=weights,
                minlength=self._spatial.refactored_nar_ids.size
                + self._spatial.refactored_cell_ids.size,
            )
        ).astype(np.float32)

        return ss.dok_array(a.reshape(-1, 1))
