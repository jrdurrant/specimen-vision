import unittest
import numpy as np
from vision.measurements.procrustes import normalise_shape


class TestProcrustes(unittest.TestCase):
    def test_normalise(self):
        """Normalising an already normalised shape should make no difference
        """
        shape = np.zeros((100, 2))
        shape[:, 0] = np.linspace(0, 2.5 * np.pi, 100)
        shape[:, 1] = np.sin(shape[:, 0])

        shape = normalise_shape(shape)
        normalised_shape = normalise_shape(shape)

        np.testing.assert_allclose(shape, normalised_shape)
