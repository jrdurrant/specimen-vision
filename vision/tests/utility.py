import cv2
import os


TEST_DATA = os.path.abspath(os.path.join(__file__, os.pardir, 'test_data'))


def get_test_image(*path):
    return cv2.imread(os.path.join(TEST_DATA, *path))
