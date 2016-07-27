import cv2
import numpy as np
from vision.image_functions import threshold
from skimage.measure import label, regionprops
from operator import attrgetter
from skimage.morphology import binary_closing


def saliency_butterfly(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return 0.25 * hsv_image[:, :, 2] + 0.75 * hsv_image[:, :, 1]


def saliency_dragonfly(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image[:, :, 1]


def crop_by_saliency(saliency_map, closing_size=11, border=50):
    binary_image = threshold(saliency_map)
    selem = np.ones((closing_size, closing_size))
    binary_image = binary_closing(binary_image, selem)

    labels = label(binary_image)
    roi = max(regionprops(labels),  key=attrgetter('filled_area'))

    border = 50
    return (slice(roi.bbox[0] - border, roi.bbox[2] + 2 * border),
            slice(roi.bbox[1] - border, roi.bbox[3] + 2 * border))
