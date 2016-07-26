from skimage.morphology import skeletonize
import cv2
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from skimage.filters import threshold_otsu
from vision.ruler_detection.hough_space import hough_transform, hspace_features
from vision.ruler_detection.find_ruler import find_ruler, best_angles
import peakutils
from scipy.ndimage.filters import gaussian_filter1d


logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


def threshold(image, mask=None):
    """Convert a full color image to a binary image

    Args:
        image (ndarray): BGR image of shape n x m x 3.
        mask (ndarray): binary image. Calculates threshold value based only on pixels within the mask.
                        However, all pixels are included in the output image.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    if mask is not None:
        input_image = image[:, :, 1]
        input_image = input_image[mask > 0].reshape(-1, 1)
    else:
        input_image = image[:, :, 1]

    threshold_val = threshold_otsu(input_image)
    return np.where(image[:, :, 1] > threshold_val, 255, 0)


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


def find_grid(hspace_angle, max_separation):
    """Returns the separation between graduations of the ruler.

    Args:
        hspace_angle: Bins outputted from :py:meth:`hough_transform`, but for only a single angle.
        max_separation: Maximum size of the *largest* graduation.
        graduations: List of graduation spacings in order of spacing size, normalised with respect to the
                     smallest graduation

    Returns:
        int: Separation between graduations in pixels

    """

    autocorrelation = acf(hspace_angle, nlags=max_separation, unbiased=False)

    smooth = gaussian_filter1d(autocorrelation, 1)
    peaks = peakutils.indexes(smooth, thres=0.25)

    # return best_separation[0][1]
    return np.mean(np.diff(np.insert(peaks[:4], 0, 0)))


def remove_large_components(binary_image, threshold_size=0):
    if threshold_size == 0:
        threshold_size = max(binary_image.shape)

    num_labels, labels = cv2.connectedComponents(binary_image.astype(np.uint8))
    sizes = np.bincount(labels.flatten())
    num_oversize = np.sum(sizes > threshold_size)
    oversize_labels = np.argsort(sizes)[:-num_oversize:-1]

    for label in oversize_labels:
        binary_image[labels == label] = 0


def ruler_scale_factor(image, graduations, distance=1):
    """Returns the scale factor to convert from image coordinates to real world coordinates

    Args:
        image: BGR image of shape n x m x 3.
        graduations: List of graduation spacings in order of spacing size. If distance is not given,
                     these will be interpreted as the actual spacings in real world coordinates.
                     Otherwise, they are interpreted as relative spacings
        distance (optional): The real world size of the smallest graduation spacing
    Returns:
        float: Unitless scale factor from image coordinates to real world coordinates.

    """

    height, width = image.shape[:2]
    image, mask = find_ruler(image)
    binary_image = mask * threshold(image, mask)

    if binary_image[mask].mean() > 128:
        binary_image[mask] = 255 - binary_image[mask]
    remove_large_components(binary_image, max(height, width))
    edges = find_edges(255 - binary_image)
    hspace, angles, distances = hough_transform(edges)
    features = hspace_features(hspace, splits=16)
    angle_index = best_angles(np.array(features))

    distance *= graduations[0]
    graduations = np.array(graduations) / graduations[0]

    max_graduation_size = int(max(image.shape))
    line_separation_pixels = find_grid(hspace[:, angle_index], max_graduation_size)

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return distance / line_separation_pixels
