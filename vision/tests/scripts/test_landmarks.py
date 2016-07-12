import cv2
import os
import unittest
import numpy as np
from vision.segmentation.segmentation import saliency_map
from vision.measurements.landmarks import shape_context, cartesian_to_polar
from vision.tests import TEST_DATA


class TestShapeContextSimple(unittest.TestCase):
    def setUp(self):
        self.n = 20

    def tearDown(self):
        pass

    def test_shape(self):
        vertices = np.random.rand(self.n, 2) * 100
        h = shape_context(vertices[0], vertices, num_bins_log_radius=5, num_bins_theta=12)
        self.assertEqual(h.shape, (5, 12))

    def test_distance(self):
        vertices = np.random.rand(self.n, 2) * 100
        h = shape_context(vertices[0], vertices, num_bins_log_radius=5, num_bins_theta=12)
        np.testing.assert_array_equal(h, h)


class TestShapeContextImage(unittest.TestCase):
    def setUp(self):
        image = cv2.imread(os.path.join(TEST_DATA, 'BMNHE_500606.JPG'))
        saliency = saliency_map(image).astype(np.uint8)
        _, binary_image = cv2.threshold(saliency, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('b.png', binary_image)
        _, contours, _ = cv2.findContours(binary_image.astype('uint8'),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        self.outline = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]

    def tearDown(self):
        pass

    def test_shape_context(self):
        vertices = self.outline[:, 0, :]
        h = shape_context(vertices[0], vertices, num_bins_log_radius=5, num_bins_theta=12)
        vertices_flipped = np.copy(vertices)
        vertices_flipped[:, 1] *= -1
        h_flipped = shape_context(vertices_flipped[0], vertices_flipped, num_bins_log_radius=5, num_bins_theta=12)
        np.testing.assert_array_equal(h, h_flipped)


class TestCoordinateConversion(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_limits(self):
        cartesian_coordinates = 100 * (np.random.rand(100, 2) - 0.5)
        x = cartesian_coordinates[:, 1]
        y = cartesian_coordinates[:, 0]
        distances, angles = cartesian_to_polar(x, y)

        self.assertGreaterEqual(np.min(distances), 0)
        self.assertGreaterEqual(np.min(angles), 0)
        self.assertLess(np.max(angles), 2 * np.pi)
