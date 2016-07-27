from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import numpy as np
import cv2


def threshold(image, mask=None):
    """Convert a full color image to a binary image

    Args:
        image (ndarray): BGR image of shape n x m x 3.
        mask (ndarray): binary image. Calculates threshold value based only on pixels within the mask.
                        However, all pixels are included in the output image.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    if len(image.shape) == 3:
        image = image[:, :, 1]

    if mask is not None:
        image = image[mask > 0]

    threshold_val = threshold_otsu(image.reshape(-1, 1))
    return np.where(image > threshold_val, 255, 0)


def find_edges(binary_ruler_image):
    """Find edges as a preprocess for line detection

    Note:
        For this purpose, only the ruler's lines are needed and so it is only in those areas that the
        edge preprocessing needs to make sense.

    Args:
        binary_ruler_image: 2D Binary image, where 0 is off and 255 is on.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    return skeletonize(1 - binary_ruler_image / 255)


def remove_large_components(binary_image, threshold_size=0):
    if threshold_size == 0:
        threshold_size = max(binary_image.shape)

    num_labels, labels = cv2.connectedComponents(binary_image.astype(np.uint8))
    sizes = np.bincount(labels.flatten())
    num_oversize = np.sum(sizes > threshold_size)
    oversize_labels = np.argsort(sizes)[:-num_oversize:-1]

    for label in oversize_labels:
        binary_image[labels == label] = 0
