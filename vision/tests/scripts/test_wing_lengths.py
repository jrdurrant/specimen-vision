import cv2
import os
import unittest
from vision.measurements.shape_analysis import wing_length
from vision.segmentation.segmentation import segment_butterfly, segment_wing
from vision.tests import TEST_DATA

class TestWingLengths500606(unittest.TestCase):
    wing_mask = None
    wing_paths = None
    centre_of_mass = None

    def setUp(self):
        image = cv2.imread(os.path.join(TEST_DATA, 'BMNHE_500606.JPG'))
        segmented_image, segmented_mask = segment_butterfly(image, saliency_threshold=96)
        self.wing_mask, self.wing_paths, self.centre_of_mass = segment_wing(segmented_mask)

    def tearDown(self):
        pass

    def test_left_wing_length(self):
        left_wing_length = wing_length(self.wing_mask[:, self.centre_of_mass[1]::-1], self.wing_paths[0][:, self.centre_of_mass[1]::-1])
        self.assertGreater(left_wing_length, 1150)
        self.assertLess(left_wing_length, 1190)

    def test_right_wing_length(self):
        right_wing_length = wing_length(self.wing_mask[:, self.centre_of_mass[1]:], self.wing_paths[1][:, self.centre_of_mass[1]:])
        self.assertGreater(right_wing_length, 1160)
        self.assertLess(right_wing_length, 1200)