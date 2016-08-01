import csv
import unittest
from nose_parameterized import parameterized
from nose.tools import nottest
from operator import itemgetter
from vision.ruler_detection.find_scale import ruler_scale_factor
from vision.tests import get_test_image, get_test_path, get_test_folder_images


class TestTransforms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        image_base = get_test_image('ruler', 'distorted', 'test.JPG')
        cls.scale_factor_base = ruler_scale_factor(image_base, distance=0.5)

    @nottest
    def generate_test_files():
        return get_test_folder_images('ruler', 'distorted')

    @parameterized.expand(generate_test_files())
    def test_transform(self, file):
        image = get_test_image(file)
        scale_factor = ruler_scale_factor(image, distance=0.5)
        self.assertAlmostEqual(self.scale_factor_base, scale_factor, delta=0.2)


class TestMeasured(unittest.TestCase):

    @nottest
    def generate_test_files():
        with open(get_test_path('ruler', 'measured', 'data.csv'), 'r') as csv_file:
            reader = csv.DictReader(csv_file, delimiter=' ')
            data = [(row['filename'], float(row['graduation_distance']))
                    for row
                    in reader]
        return sorted(data, key=itemgetter(0))

    @parameterized.expand(generate_test_files())
    def test_measurement(self, file, separation):
        image = get_test_image('ruler', 'measured', file)
        scale_factor = ruler_scale_factor(image, distance=0.5)
        self.assertAlmostEqual(separation, 0.5 / scale_factor, delta=1)
