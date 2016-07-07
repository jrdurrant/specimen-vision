import cv2
import os
import fnmatch
import unittest
from nose_parameterized import parameterized
from nose.tools import nottest
from vision.measurements.detect_ruler import ruler_line_separation
from vision.tests import TEST_DATA


class TestTransforms(unittest.TestCase):
    separation_base = 0
    image_base = None
    filenames = None

    def setUp(self):
        self.image_base = cv2.imread(os.path.join(TEST_DATA, 'ruler', 'test.JPG'))
        self.scale_factor_base = ruler_line_separation(self.image_base)

    def tearDown(self):
        pass

    @nottest
    def generate_test_files():
        test_dir_files = os.listdir(os.path.join(TEST_DATA, 'ruler'))
        return sorted(fnmatch.filter(test_dir_files, '*.JPG'))

    @parameterized.expand(generate_test_files())
    def test_transform(self, file):
        image = cv2.imread(os.path.join(TEST_DATA, 'ruler', file))
        separation = ruler_line_separation(image)
        self.assertAlmostEqual(self.scale_factor_base, separation, delta=0.2)
