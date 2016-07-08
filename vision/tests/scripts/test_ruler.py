import cv2
import os
import fnmatch
import unittest
from nose_parameterized import parameterized
from nose.tools import nottest
from vision.measurements.detect_ruler import ruler_scale_factor
from vision.measurements.detect_ruler import remove_multiples
from vision.tests import TEST_DATA


class TestTransforms(unittest.TestCase):
    scale_factor_base = 0
    image_base = None
    filenames = None

    def setUp(self):
        self.image_base = cv2.imread(os.path.join(TEST_DATA, 'ruler', 'test.JPG'))
        self.scale_factor_base = ruler_scale_factor(self.image_base)

    def tearDown(self):
        pass

    @nottest
    def generate_test_files():
        test_dir_files = os.listdir(os.path.join(TEST_DATA, 'ruler'))
        return sorted(fnmatch.filter(test_dir_files, '*.JPG'))

    @parameterized.expand(generate_test_files())
    def test_transform(self, file):
        image = cv2.imread(os.path.join(TEST_DATA, 'ruler', file))
        scale_factor = ruler_scale_factor(image)
        self.assertAlmostEqual(self.scale_factor_base, scale_factor, delta=0.2)


class TestRemoveMultiples(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_evens(self):
        evens = range(2, 10, 2)
        evens_no_multiples = remove_multiples(evens)
        self.assertEqual(evens_no_multiples, [2])
