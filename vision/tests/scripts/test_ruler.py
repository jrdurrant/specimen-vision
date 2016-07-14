import csv
import unittest
import numpy as np
from nose_parameterized import parameterized
from nose.tools import nottest
from operator import itemgetter
from vision.measurements.detect_ruler import ruler_scale_factor
from vision.measurements.detect_ruler import remove_multiples
from vision.tests import get_test_image, get_test_path, get_test_folder_images


class TestTransforms(unittest.TestCase):
    scale_factor_base = 0

    @classmethod
    def setUpClass(self):
        image_base = get_test_image('ruler', 'distorted', 'test.JPG')
        self.scale_factor_base = ruler_scale_factor(image_base, graduations=[1, 2, 20], distance=0.5)

    @nottest
    def generate_test_files():
        return get_test_folder_images('ruler', 'distorted')

    @parameterized.expand(generate_test_files())
    def test_transform(self, file):
        image = get_test_image(file)
        scale_factor = ruler_scale_factor(image, graduations=[1, 2, 20], distance=0.5)
        self.assertAlmostEqual(self.scale_factor_base, scale_factor, delta=0.2)


class TestMeasured(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    @nottest
    def generate_test_files():
        with open(get_test_path('ruler', 'measured', 'data.csv'), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            data = [(filename, float(separation)) for filename, separation in reader]
        return sorted(data, key=itemgetter(0))

    @parameterized.expand(generate_test_files())
    def test_measurement(self, file, separation):
        image = get_test_image('ruler', 'measured', file)
        scale_factor = ruler_scale_factor(image, graduations=[1, 2, 20], distance=0.5)
        self.assertAlmostEqual(separation, 0.5 / scale_factor, delta=0.5)


class TestRemoveMultiples(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_evens(self):
        evens = list(zip(range(2, 10, 2), range(2, 10, 2)))
        evens_no_multiples = remove_multiples(evens)
        self.assertEqual(evens_no_multiples, [(2, 2)])

    def test_evens_reverse(self):
        evens = list(zip(range(2, 10, 2), range(2, 10, 2)[::-1]))
        evens_no_multiples = remove_multiples(evens)
        self.assertEqual(evens_no_multiples, [(8, 2)])

    def test_random(self):
        indices = (np.random.rand(20) * 0.1 - 0.05) + np.random.randint(1, 10, 20)
        indices[0] = 1
        scores = list(zip(np.random.rand(10) * 10, indices))
        scores_no_multiples = remove_multiples(scores)
        max_ratio = max(scores_no_multiples, key=itemgetter(1))[1]
        min_ratio = min(scores_no_multiples, key=itemgetter(1))[1]
        self.assertLessEqual(abs(max_ratio - 1), 1 + 0.05)
        self.assertLessEqual(abs(min_ratio - 1), 1 + 0.05)
