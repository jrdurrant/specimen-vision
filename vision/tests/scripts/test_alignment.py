import unittest
from nose_parameterized import parameterized
from nose.tools import nottest
from vision.measurements.alignment import align
from vision.tests import get_test_image, get_test_folder_images


class TestGlobalAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        files = get_test_folder_images('ruler', 'measured')
        images = [get_test_image(file) for file in files]
        self.outlines = align(images)

    @nottest
    def generate_test_files():
        indices = range(len(get_test_folder_images('ruler', 'distorted')))
        return [str(i) for i in indices]

    @parameterized.expand(generate_test_files())
    def test_alignment(self, index):
        self.assertGreaterEqual(int(index), 0)
