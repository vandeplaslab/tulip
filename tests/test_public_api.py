import numpy as np

from tulip import MIX, NNLS, SPATIAL, SPECTRAL


def test_public_api_and_in_memory_example_data():
    """The documented imports and mock-data workflow should remain usable."""
    view = np.zeros((16, 16), dtype=np.int32)
    view[2:8, 2:8] = 1
    view[8:14, 8:14] = 2

    spatial = SPATIAL(nar_ids=np.array([0]), cut_off=1)
    spatial.from_array(view)
    spectral = SPECTRAL(
        cell=np.array([[1.0, 0.5], [0.5, 1.0]]),
        nar=np.array([[0.1, 0.1]]),
        cell_distribution=np.array([0.5, 0.5]),
        nar_distribution=np.array([1.0]),
    )

    mixed = MIX(spatial, spectral)
    mixed.link()
    mixed.downsample(4)
    solution = NNLS(mixed.overlap, mixed.mixed)
    solution.run()

    assert solution.c.shape == mixed.spectra.shape
    assert mixed.class_image.shape == view.shape
