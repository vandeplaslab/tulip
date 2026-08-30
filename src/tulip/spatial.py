import numpy as np
from tifffile import TiffFile


class SPATIAL:
    """
    Generate and manage synthetic cell spatial views from image data.

    This class handles the processing of cellular spatial data, particularly from TIFF files,
    managing cell IDs, and handling non-annotated regions (nar).

    Parameters
    ----------
    view : np.ndarray, optional
        Preloaded spatial view of cells. Default is None.
    nar_ids : np.ndarray, optional
        IDs for non-annotated regions. Default is np.zeros(1).
    cut_off : int, optional
        Minimum size threshold for cells. Cells smaller than this are removed. Default is 50.

    Attributes
    ----------
    view : np.ndarray
        2D array containing the spatial view of cells
    nar_ids : np.ndarray
        Array of IDs representing non-annotated regions
    cut_off : int
        Size threshold for filtering small cells
    cell_ids : np.ndarray
        Array of original cell IDs
    refactored_cell_ids : np.ndarray
        Array of remapped cell IDs
    refactored_nar_ids : np.ndarray
        Array of remapped non-annotated region IDs
    viewport : dict
        Dictionary containing view boundaries (x_min, x_max, y_min, y_max)
    """

    def __init__(
        self,
        view: np.ndarray = None,
        nar_ids: np.ndarray = np.zeros(1),
        cut_off: int = 50,
    ):
        """
        Initialize the spatial class.

        Parameters
        ----------
        view : np.ndarray, optional
            Preloaded spatial view of cells. Default is None.
        nar_ids : np.ndarray, optional
            IDs for non-annotated regions. Default is np.zeros(1).
        cut_off : int, optional
            Minimum size threshold for cells. Default is 50.
        """
        self.view = view
        self.nar_ids = nar_ids
        self.cut_off = cut_off
        self.cell_ids = None
        self.refactored_cell_ids = None
        self.refactored_nar_ids = None
        self.viewport = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}

    @property
    def size(self) -> tuple:
        """
        Get the dimensions of the spatial view.

        Returns
        -------
        tuple
            A tuple containing the height and width of the spatial view.
        """
        return self.view.shape

    @property
    def nar_num(self) -> int:
        """
        Get the number of non-annotated regions.

        Returns
        -------
        int
            The total number of non-annotated regions.
        """
        return self.refactored_nar_ids.size

    @property
    def cell_num(self) -> int:
        """
        Get the number of unique cells.

        Returns
        -------
        int
            The total number of unique cells in the view.
        """
        return self.refactored_cell_ids.size

    @property
    def pixel_num(self) -> int:
        """
        Get the total number of pixels in the view.

        Returns
        -------
        int
            The total number of pixels in the spatial view.
        """
        return np.prod(self.view.shape)

    def from_tiff(
        self,
        tiff_file: str,
        x_min: int,
        y_min: int,
        x_size: int,
        y_size: int,
        page: int = 0,
    ) -> None:
        """
        Generate spatial view from a TIFF file by cropping a specific region.

        This method reads a TIFF file, crops it to the specified dimensions,
        removes small cells, and sets up the cell ID mapping system.

        Parameters
        ----------
        tiff_file : str
            Path to the TIFF file
        x_min : int
            X-coordinate of lower-left corner of crop
        y_min : int
            Y-coordinate of lower-left corner of crop
        x_size : int
            Width of the crop
        y_size : int
            Height of the crop
        page : int, optional
            Page number in TIFF file to process. Default is 0.

        Notes
        -----
        This method updates several instance attributes including view, cell_ids,
        and refactored_cell_ids.
        """
        crop = self._read_tiff(tiff_file, page, x_min, y_min, x_size, y_size)
        self.from_array(crop)

    def from_array(self, view: np.ndarray) -> None:
        """Load a labelled spatial view from an in-memory array.

        This is useful for synthetic examples and for data that has already been
        read from a non-TIFF source.  Values in ``nar_ids`` are treated as
        non-annotated regions; all other labels are treated as cell IDs.
        """
        if view.ndim != 2:
            raise ValueError("view must be a two-dimensional labelled array")

        view = np.asarray(view, dtype=np.int32).copy()
        view = self._cut_off(view)
        self._find_cell_ids(view)
        self._define_refactor_ids(0, self.cell_ids.size)
        self.view = self._refactor_ids(view)
        self._empty_check()

    def _empty_check(self) -> None:
        """
        Check if the view contains any cells and warn if empty.

        This method prints a warning message if no cells are detected in the current view.

        Returns
        -------
        None
        """
        if self.cell_num == 0:
            print("Warning: there are no cells in this view")

    def _cut_off(self, crop: np.ndarray) -> np.ndarray:
        """
        Remove small objects (cells) below the cut_off threshold.

        Parameters
        ----------
        crop : np.ndarray
            Input image array containing cell IDs

        Returns
        -------
        np.ndarray
            Processed image with small cells replaced by NAR IDs

        Notes
        -----
        Cells with pixel counts less than or equal to self.cut_off are replaced
        with non-annotated region IDs.
        """
        unique, unique_counts = np.unique(crop.flatten(), return_counts=True)
        for i in np.where(unique_counts <= self.cut_off)[0]:
            crop[crop == unique[i]] = self.nar_ids

        print(
            "Removed "
            + str(len(np.where(unique_counts <= self.cut_off)[0]))
            + " Cells"
        )
        return crop

    def _define_refactor_ids(
        self,
        min_id: int,
        max_id: int,
    ) -> None:
        """
        Create new sequential IDs for cells and non-annotated regions.

        Parameters
        ----------
        min_id : int
            Starting ID for cell remapping
        max_id : int
            Ending ID for cell remapping

        Notes
        -----
        Updates refactored_cell_ids and refactored_nar_ids with sequential
        integer IDs. NAR IDs start after the highest cell ID.
        """
        self.refactored_cell_ids = np.linspace(
            min_id, max_id - 1, max_id, dtype=np.int32
        )
        self.refactored_nar_ids = (
            max_id
            - 1
            + np.linspace(
                1, self.nar_ids.size, self.nar_ids.size, dtype=np.int32
            )
        )

    def _find_cell_ids(self, crop: np.ndarray) -> None:
        """
        Extract unique cell IDs from the image, excluding NAR IDs.

        Parameters
        ----------
        crop : np.ndarray
            Input image array containing cell and NAR IDs

        Notes
        -----
        Updates the cell_ids attribute with unique cell IDs found in the image,
        excluding any IDs that match non-annotated region IDs.
        """
        cell_ids = np.unique(crop)
        self.cell_ids = np.setdiff1d(cell_ids, self.nar_ids)

    def _refactor_ids(
        self,
        crop: np.ndarray,
    ) -> np.ndarray:
        """
        Remap cell and NAR IDs to new sequential IDs.

        Parameters
        ----------
        crop : np.ndarray
            Input image array with original IDs

        Returns
        -------
        np.ndarray
            Image array with remapped sequential IDs

        Notes
        -----
        Creates a new mapping where both cell IDs and NAR IDs are sequential,
        with NAR IDs starting after the highest cell ID.
        """
        crop_copy = crop.astype(np.int64, copy=True)
        for i, e in zip(self.cell_ids, self.refactored_cell_ids):
            crop[crop_copy == i] = e
        for i, e in zip(self.nar_ids, self.refactored_nar_ids):
            crop[crop_copy == i] = e

        return crop

    def _set_viewport(
        self,
        viewport: list,
    ) -> None:
        """
        Set the viewport boundaries.

        Parameters
        ----------
        viewport : list
            List of [x_min, x_max, y_min, y_max] coordinates defining the
            boundaries of the current view

        Notes
        -----
        Updates the viewport dictionary with the new boundary coordinates.
        """
        self.viewport = {
            "x_min": viewport[0],
            "x_max": viewport[1],
            "y_min": viewport[2],
            "y_max": viewport[3],
        }

    def _read_tiff(
        self,
        tiff_file: str,
        page: int,
        x_min: int,
        y_min: int,
        x_size: int,
        y_size: int,
    ) -> np.ndarray:
        """
        Read and crop a region from a TIFF file.

        Parameters
        ----------
        tiff_file : str
            Path to TIFF file
        page : int
            Page number to read from the TIFF file
        x_min : int
            Starting x-coordinate for crop
        y_min : int
            Starting y-coordinate for crop
        x_size : int
            Width of the crop
        y_size : int
            Height of the crop

        Returns
        -------
        np.ndarray
            Cropped image data as 32-bit integers

        Notes
        -----
        Also updates the viewport boundaries based on the crop coordinates.
        """
        with TiffFile(tiff_file) as tif:
            crop = self._get_crop(tif.pages[page], y_min, x_min, y_size, x_size)

        self._set_viewport([x_min, x_min + x_size, y_min, y_min + y_size])

        return crop.astype(np.int32, copy=True)

    def _get_crop(self, page, i0: int, j0: int, h: int, w: int) -> np.ndarray:
        """
        Extract a crop from a tiled TIFF image efficiently.

        Only loads the tiles that contain the desired crop area. Adapted from:
        https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743

        Parameters
        ----------
        page : TiffPage
            TIFF page object to extract from
        i0 : int
            Starting row coordinate (top-left corner)
        j0 : int
            Starting column coordinate (top-left corner)
        h : int
            Height of desired crop
        w : int
            Width of desired crop

        Returns
        -------
        np.ndarray
            Cropped image data

        Raises
        ------
        ValueError
            If the page is not tiled, if crop dimensions are invalid,
            or if crop area is outside image bounds.
        """
        if not page.is_tiled:
            # For non-tiled images, read the entire image and crop manually
            crop = page.asarray()[i0 : i0 + h, j0 : j0 + w]
            return crop

        im_width = page.imagewidth
        im_height = page.imagelength

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
            raise ValueError("Requested crop area is out of image bounds.")

        tile_width, tile_height = page.tilewidth, page.tilelength
        i1, j1 = i0 + h, j0 + w
        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(
            int
        )
        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.empty(
            (
                (tile_i1 - tile_i0) * tile_height,
                (tile_j1 - tile_j0) * tile_width,
            ),
            dtype=page.dtype,
        )

        fh = page.parent.filehandle
        jpegtables = page.tags.get("JPEGTables", None)
        if jpegtables is not None:
            jpegtables = jpegtables.value

        for i in range(tile_i0, tile_i1):
            for j in range(tile_j0, tile_j1):
                index = int(i * tile_per_line + j)
                offset = page.dataoffsets[index]
                bytecount = page.databytecounts[index]

                fh.seek(offset)
                data = fh.read(bytecount)
                tile, indices, shape = page.decode(data, index)

                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[
                    im_i : im_i + tile_height, im_j : im_j + tile_width
                ] = np.squeeze(tile)

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        return out[im_i0 : im_i0 + h, im_j0 : im_j0 + w]

    def split_nar(
        self,
        scale: int,
    ) -> None:
        """
        Split non-annotated regions into grid-based segments.

        This method divides non-annotated regions into a grid pattern, assigning
        new IDs to each grid cell.

        Parameters
        ----------
        scale : int
            Size of grid cells for splitting NAR regions. This determines the
            granularity of the grid pattern.

        Notes
        -----
        Updates the view with new NAR IDs organized in a grid pattern.
        The new NAR IDs start after the highest cell ID.
        """
        num = self.view.shape[0] // scale
        vec = max(self.refactored_cell_ids) + np.linspace(
            1, num**2, num**2, dtype=int
        )
        vec2 = vec.reshape(num, num)
        tile = np.repeat(vec2, scale, axis=0)
        tile = np.repeat(tile, scale, axis=1)
        self.view = (
            self.view * (np.isin(self.view, self.refactored_cell_ids))
            + (np.isin(self.view, self.refactored_nar_ids)) * tile
        )
        self.refactored_nar_ids = vec
