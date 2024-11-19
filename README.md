# TULIP
specTrometric Unmixing of single-ceLL by Inverse Problem

<img src="https://github.com/RogerMoens/tulip/blob/master/example/overlap.png" alt="Nomenclature" width="400"/>

**An example Jupyter Notebook is available from example.ipynb**

**1. Generate spatial view**
```python
from tulip import SPATIAL
a = SPATIAL()
a.from_tiff("path/to/file.tiff", 1000, 1000, 100, 100) # x, y, x_size, y_size
```
```python
# Print all class variables
print(vars(a))
```

```python
# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))
plt.imshow(a.view, vmin=0, origin='lower')
plt.colorbar()
plt.show()
```

**2. Generate spectral view**
```python
from tulip import SPECTRAL
b = SPECTRAL()
# Generate spectra from condition on first input (sparse_overlap_matrix) for cell and for nar
# Alternatively, one can also provide spectra for nar and cell through the initialization
b.from_real(sparse_overlap_matrix, spectra, 'a >= 0.99999', 10, annotation = 'cell')
b.from_real(sparse_overlap_matrix, spectra, 'a.sum(axis=1) < 0.00001', 1, annotation = 'nar')
```
```python
# Print all class variables: cell spectra, nar spectra, cell class distribution, nar class distribution, number of different cell spectra, number of m/z bins
print(b.cell, b.nar, b.cell_distribution, b.nar_distribution, b.cell_num, b.spec_num)
```

**3. Generate synthetic data set**
```python
from tulip as MIX
c = MIX(a, b) # Provide tulip.spatial and tulip.spectral instances earlier created
c.link() # Link spectra to cells: (1) sample classes of spectra and link to spatial cells/nar; (2) generate a sparse matrix that links the spectra to cells/nar
c.downsample(16, type='linear') # Downsample linearly with a scale of 16 (as provided by Heath)
```
```python
# Y = V * W
# Y : c.mixed (2-dimensional, flattened) or c.mixed_block (3-dimensional)
# V : c.overlap
# W : c.spectra

from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(1, 2, figsize=(20, 10))
im = axs[0].imshow(c.mixed_block.sum(axis=2), aspect='equal')

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')
axs[0].set_title('Total Ion Image')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')

axs[1].plot(c.mixed_block.sum(axis=(0, 1)))
axs[1].grid()
axs[1].set_title('Total Ion Count')
axs[1].set_xlabel('m/z bin')
axs[1].set_ylabel('Ion Count (a.u.)')

plt.show()
```

```python
# Plot image of spatial view where the value of cells equals their class
plt.imshow(c.class_image)
plt.colorbar()
plt.show()
```

**3. Unmix tulipetic data set**
```python
from tulip import LS
u = LS(c.overlap, c.mixed)
mat = u.run()

from tulip import NNLS
u = NNLS(c.overlap, c.mixed)
mat = u.run()

from tulip import SVT
u = SVT(c.overlap, c.mixed)
mat = u.run()

from tulip import tulip
u = tulip(c.overlap, c.mixed)
mat = u.run()
```
