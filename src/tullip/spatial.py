import numpy as np
from tifffile import TiffFile

# to do :
# - Check _get_crop whether this does not read things into memory we don't want
# - finish commenting
# - finish refactored nar ids


class spatial:
    """
    Generate synthetic cell spatial view:

    Parameters
    ----------
    view: default=None
        Preload a view into the class

    Attributes
    ----------


    """

    def __init__(
        self,
        view: np.ndarray = None,
        nar_ids: np.ndarray = np.zeros(1),
        cut_off: int = 50,
    ):
        self.view = view
        self.nar_ids = nar_ids
        self.cut_off = cut_off
        # Other attributes
        self.cell_ids = None
        self.refactored_cell_ids = None
        self.refactored_nar_ids = None
        self.viewport = {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0}

        pass

    @property
    def size(self) -> tuple:
        return self.view.shape

    @property
    def nar_num(self) -> int:
        return self.refactored_nar_ids.size

    @property
    def cell_num(self) -> int:
        return self.refactored_cell_ids.size

    @property
    def pixel_num(self) -> int:
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
        """Generate spatial view from TIFF file

        Parameters
        ----------
        tiff_file :
            String containing path to the TIFF file
        x_min :
            Pixel coordinate along x-axis corresponding to the lower left vertex of square/rectangle crop
        y_min :
            Pixel coordinate along y-axis corresponding to the lower left vertex of square/rectangle crop
        x_size :
            Width of the crop along x-axis
        y_size :
            Height of the crop along y-axis
        page : default=0
            TIFF page with to be cropped image
        """
        crop = self._read_tiff(
            tiff_file, page, x_min, y_min, x_size, y_size
        )  # get crop
        crop = self._cut_off(crop) # remove very small cells
        self._find_cell_ids(crop)  # get cell ids from crop
        self._define_refactor_ids(
            0, self.cell_ids.size 
        )  # define new refactored cell  
        self.view = self._refactor_ids(crop)  # create crop (view) with new cell ids
        self._empty_check()  # check if view is empty

        return None

    def _empty_check(self) -> None:
        """

        Parameters
        ----------

        """
        if self.cell_num == 0:
            print("Warning: there are no cells in this view")

        return None
    
    def _cut_off(self, crop) -> np.ndarray:
        """

        Parameters
        ----------

        """
        unique, unique_counts = np.unique(crop.flatten(), return_counts=True)
        for i in np.where(unique_counts <= self.cut_off)[0]:
            crop[crop == unique[i]] = self.nar_ids
        
        print('Removed '+str(len(np.where(unique_counts <= self.cut_off)[0]))+' Cells')
        return crop

    def _define_refactor_ids(
        self,
        min_id: int,
        max_id: int,
    ) -> None:
        """Create new refactoring for cell ids

        Parameters
        ----------
        min_id :

        max_id :

        Returns
        -------

        """
        self.refactored_cell_ids = np.linspace(
            min_id, max_id-1, max_id, dtype=np.int32
        )
        self.refactored_nar_ids = max_id-1+np.linspace(
            1, self.nar_ids.size, self.nar_ids.size, dtype=np.int32
        )

        return None

    def _find_cell_ids(self, crop: np.ndarray) -> None:
        cell_ids = np.unique(crop)  # select unique cell ids
        self.cell_ids = np.setdiff1d(cell_ids, self.nar_ids)  # remove nar ids

        return None

    def _refactor_ids(
        self,
        crop: np.ndarray,
    ) -> np.ndarray:
        crop_copy = crop.astype(np.int64, copy=True)  # copy to generate reference
        for i, e in zip(self.cell_ids, self.refactored_cell_ids):  # refactor
            crop[crop_copy == i] = e
        for i, e in zip(self.nar_ids, self.refactored_nar_ids):  # refactor
            crop[crop_copy == i] = e

        return crop

    def _set_viewport(
        self,
        viewport: list,
    ) -> None:
        self.viewport = {
            "x_min": viewport[0],
            "x_max": viewport[1],
            "y_min": viewport[2],
            "y_max": viewport[3],
        }

        return None

    def _read_tiff(
        self,
        tiff_file: str,
        page: int,
        x_min: int,
        y_min: int,
        x_size: int,
        y_size: int,
    ) -> np.ndarray:
        """Read part of image in TIFF file on specific page"""
        with TiffFile(tiff_file) as tif:
            crop = self._get_crop(tif.pages[page], y_min, x_min, y_size, x_size)

        self._set_viewport(
            [x_min, x_min + x_size, y_min, y_min + y_size]
        )  # set viewport

        return crop.astype(np.int32, copy=True)

    def _get_crop(self, page, i0: int, j0: int, h: int, w: int) -> np.ndarray:
        """Extract a crop from a TIFF image file directory (IFD).

        Only the tiles englobing the crop area are loaded and not the whole page.
        This is usefull for large Whole slide images that can't fit int RAM.

        # Adapted from: https://gist.github.com/rfezzani/b4b8852c5a48a901c1e94e09feb34743

        Parameters
        ----------
        page : TiffPage
            TIFF image file directory (IFD) from which the crop must be extracted.
        i0, j0: int
            Coordinates of the top left corner of the desired crop.
        h: int
            Desired crop height.
        w: int
            Desired crop width.
        Returns
        -------
        out : ndarray of shape (imagedepth, h, w, sampleperpixel)
            Extracted crop.
        """

        if not page.is_tiled:
            raise ValueError("Input page must be tiled.")

        im_width = page.imagewidth
        im_height = page.imagelength

        if h < 1 or w < 1:
            raise ValueError("h and w must be strictly positive.")

        if i0 < 0 or j0 < 0 or i0 + h >= im_height or j0 + w >= im_width:
            raise ValueError("Requested crop area is out of image bounds.")

        tile_width, tile_height = page.tilewidth, page.tilelength
        i1, j1 = i0 + h, j0 + w

        tile_i0, tile_j0 = i0 // tile_height, j0 // tile_width
        tile_i1, tile_j1 = np.ceil([i1 / tile_height, j1 / tile_width]).astype(int)

        tile_per_line = int(np.ceil(im_width / tile_width))

        out = np.empty(
            ((tile_i1 - tile_i0) * tile_height, (tile_j1 - tile_j0) * tile_width),
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
                tile, indices, shape = page.decode(data, index)#, jpegtables)

                im_i = (i - tile_i0) * tile_height
                im_j = (j - tile_j0) * tile_width
                out[im_i : im_i + tile_height, im_j : im_j + tile_width] = np.squeeze(
                    tile
                )

        im_i0 = i0 - tile_i0 * tile_height
        im_j0 = j0 - tile_j0 * tile_width

        return out[im_i0 : im_i0 + h, im_j0 : im_j0 + w]

    def split_nar(
        self,
        scale: int,
    ) -> None:
        """

        Parameters
        ----------

        Returns
        -------

        """
        num = self.view.shape[0]//scale
        vec = max(self.refactored_cell_ids)+np.linspace(1, num**2, num**2, dtype=int)
        vec2 = vec.reshape(num, num)
        tile = np.repeat(vec2, scale, axis=0)
        tile = np.repeat(tile, scale, axis=1)    
        self.view = self.view*(np.isin(self.view, self.refactored_cell_ids))+(np.isin(self.view, self.refactored_nar_ids))*tile
        self.refactored_nar_ids = vec
        
        return None
    
    # def remove_nar(self) -> None:
    #     loc = np.where(self.view == self.refactored_nar_ids)
    #     self.view[loc] = 88888
    #     self.refactored_nar_ids = np.array([])
    #     return None