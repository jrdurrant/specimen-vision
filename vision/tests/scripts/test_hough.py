import unittest
import numpy as np
from scipy.stats import entropy
from nose_parameterized import parameterized
from nose.tools import nottest
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
    def setUp(self):
        # Number of random arrays to test
        self.n = 100

    @nottest
    def generate_test_files():
        return ['{:02d}'.format(i) for i in range(5, 100)]

    @parameterized.expand(generate_test_files())
    def test_vs_reference(self, length):
        array_length = int(length)
        array = np.random.randint(0, 100000, array_length)
        self.assertAlmostEquals(average_local_entropy(array), average_local_entropy_reference(array))
