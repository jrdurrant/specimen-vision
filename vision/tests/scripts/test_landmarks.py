import cv2
import os
import unittest
import numpy as np
from vision.segmentation.segmentation import saliency_map, largest_components
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
        self.outline = largest_components(binary_image, num_components=1)[0]

    def tearDown(self):
        pass

    def test_shape_context(self):
        image_flipped = cv2.imread(os.path.join(TEST_DATA, 'BMNHE_500606_flipped.JPG'))
        saliency = saliency_map(image_flipped).astype(np.uint8)
        _, binary_image = cv2.threshold(saliency, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        outline_flipped = largest_components(binary_image, num_components=1)[0]
        cv2.imwrite('b.png', self.outline.draw(image=np.zeros_like(binary_image), filled=True))
        cv2.imwrite('b2.png', outline_flipped.draw(image=np.zeros_like(binary_image), filled=True))
        vertices = self.outline.points[:, 0, :]
        vertices_flipped = outline_flipped.points[:, 0, :]
        h = shape_context((2436, 1457),
                          vertices,
                          num_bins_log_radius=5,
                          num_bins_theta=12)
        h_flipped = shape_context((4339, 1529),
                                  vertices_flipped,
                                  num_bins_log_radius=5,
                                  num_bins_theta=12)
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
