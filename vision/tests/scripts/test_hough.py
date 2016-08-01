import unittest
import numpy as np
from scipy.stats import entropy
from vision.ruler_detection.hough_space import average_local_entropy


def average_local_entropy_reference(arr, window_size=2):
    """Calculate the average entropy computed with a sliding window.

    Note:
        This should behave exactly the same as
        :py:meth:`vision.ruler_detection.hough_space.average_local_entropy`

        It is more intuitive but performs much slower and so is here as a reference implementation

    """
    length = arr.size
    return (1 / length) * sum(entropy(arr[(i - window_size):(i + window_size + 1)])
                              for i
                              in range(window_size, length - window_size))


class TestCorrectness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use a known seed so that the 'random' arrays are the same every time the test is run
        np.random.seed(14)

    def test_vs_reference_100(self):
        for length in range(5, 105):
            with self.subTest(length=length):
                array = np.random.randint(0, 100000, length)
                self.assertAlmostEquals(average_local_entropy(array), average_local_entropy_reference(array))
