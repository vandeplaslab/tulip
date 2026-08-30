# TULIP :tulip:
The in-silico Unmixing of singLe-cell spectra by an Inverse Problem for imaging mass spectrometry

<img src="https://github.com/vandeplaslab/tulip/blob/master/example/overlap.png" alt="Nomenclature" width="400"/>

The self-contained [example notebook](example/example.ipynb) uses the mock
dataset in `example/data/mock_tulip_data.npz`; it does not require local or
private data files.

## Installation

```bash
pip install .
```

For development from a checkout, use `pip install -e .`. The package exposes
the solver and data classes directly from `tulip` (for example,
`from tulip import MIX, NNLS, SPATIAL, SPECTRAL`); the corresponding source
module filenames are lowercase.

**1. Generate spatial view**
```python
from tulip import SPATIAL

spatial = SPATIAL()
spatial.from_tiff("path/to/file.tiff", 1000, 1000, 100, 100)
```
```python
# Print all class variables
print(vars(spatial))
```

```python
# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.imshow(spatial.view, vmin=0, origin='lower')
plt.colorbar()
plt.show()
```

**2. Generate spectral view**
```python
from tulip import SPECTRAL

spectral = SPECTRAL()
# Generate spectra from condition on first input (sparse_overlap_matrix) for cell and for nar
# Alternatively, one can also provide spectra for nar and cell through the initialization
spectral.from_real(sparse_overlap_matrix, spectra, 'a >= 0.99999', 10, annotation='cell')
spectral.from_real(sparse_overlap_matrix, spectra, 'a.sum(axis=1) < 0.00001', 1, annotation='nar')
```
```python
# Print all class variables: cell spectra, nar spectra, cell class distribution, nar class distribution, number of different cell spectra, number of m/z bins
print(spectral.cell, spectral.nar, spectral.cell_distribution, spectral.nar_distribution, spectral.cell_num, spectral.spec_num)
```

**3. Generate synthetic data set**
```python
from tulip import MIX

mixed_data = MIX(spatial, spectral)
mixed_data.link()  # Link spectra to cells and non-annotated regions.
mixed_data.downsample(16, dtype='linear')
```
```python
# Y = V * W
# Y : mixed_data.mixed (2-dimensional, flattened) or mixed_data.mixed_block (3-dimensional)
# V : mixed_data.overlap
# W : mixed_data.spectra

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
im = axs[0].imshow(mixed_data.mixed_block.sum(axis=2), aspect='equal')

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
axs[0].set_title('Total Ion Image')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].plot(mixed_data.mixed_block.sum(axis=(0, 1)))
axs[1].grid()
axs[1].set_title('Total Ion Count')
axs[1].set_xlabel('m/z bin')
axs[1].set_ylabel('Ion Count (a.u.)')

plt.show()
```

```python
# Plot image of spatial view where the value of cells equals their class
plt.imshow(mixed_data.class_image)
plt.colorbar()
plt.show()
```

**4. Unmix the synthetic data set**
```python
from tulip import LS

solution = LS(mixed_data.overlap, mixed_data.mixed)
solution.run()

from tulip import NNLS

solution = NNLS(mixed_data.overlap, mixed_data.mixed)
solution.run()
```

`LS`, `NNLS`, `SVT`, and `TULIP` are solver classes. There is no `unmix`
module; instantiate the solver you intend to use with the overlap matrix and
the mixed data, then call `.run()`.
