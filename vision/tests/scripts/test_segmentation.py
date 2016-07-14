import unittest
from vision.segmentation.segmentation import segment_butterfly
from vision.tests import get_test_image


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_segment_500606(self):
        image = get_test_image('BMNHE_500606.JPG')
        segmented_image, segmented_mask = segment_butterfly(image, saliency_threshold=64)
        segmented_image_height, segmented_image_width = segmented_image.shape[:2]
        segmented_mask_height, segmented_mask_width = segmented_mask.shape[:2]

        self.assertEqual(segmented_image.shape[:2], segmented_mask.shape[:2])
        self.assertAlmostEqual(segmented_image_width, 2575, delta=75)
        self.assertAlmostEqual(segmented_image_height, 1575, delta=75)
