from skimage.morphology import skeletonize
import cv2
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from operator import itemgetter
from vision.segmentation.segmentation import largest_components
from vision import Ruler
from sklearn.cluster import KMeans
from scipy.ndimage import find_objects
from skimage.filters import threshold_otsu
from vision.ruler_detection.hough_space import hough_transform, hspace_features
from vision.ruler_detection.find_ruler import find_ruler, best_angles

import matplotlib.pyplot as plt


logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


def remove_blocks(binary_image, min_size=10):
    def block_score(sat, size):
        return sat[size:, size:] + sat[:-size, :-size] - sat[size:, :-size] - sat[:-size, size:]

    block_size = 2 * min_size + 1
    summed_area_table = np.cumsum(np.cumsum(binary_image, axis=0), axis=1).astype(np.float64)
    score = block_score(summed_area_table, block_size) / (block_size**2)
    score = (score > 0.65) * 1.0
    inside_block = np.cumsum(np.cumsum(score[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    blocks = (block_score(inside_block, block_size) > 0) * 1.0

    binary_image_blocks_removed = np.copy(binary_image)
    binary_image_blocks_removed[(2 * block_size):, (2 * block_size):] -= blocks
    return np.clip(binary_image_blocks_removed, 0, 1)


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


def fill_gaps(edges, iterations=1):
    """Fill in *small* gaps in structures of a binary image

    Args:
        edges: 2D Binary image, where 0 is off and 255 is on.
        iterations: Number of iterations. Higher numbers of iterations fills larger gaps, but can cause
                    thin structures to collapse and merge together.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    edges = edges * 1.0
    kernel = np.ones((3, 3))
    for i in range(iterations):
        edges = cv2.dilate(edges, kernel)
        edges = skeletonize(edges) * 1.0
    return edges * 255.0


def remove_multiples(scores, ratios, threshold=0.05):
    """Returns a list where no elements are integer multiples of any other elements

    """
    n = len(scores)

    scores.sort(key=itemgetter(1))

    filtered_scores = []
    for i in range(n):
        multiple = False
        for j in range(i):
            ratio = scores[i][1] / scores[j][1]
            # hack. not sure ratio actually matters
            for r in [1]:
                remainder = abs(ratio % r)
                if remainder > (r - 0.5):
                    remainder = abs(remainder - r)
                if ratio > (1 + 2 * threshold) and remainder < threshold:
                    multiple = True
        if not multiple:
            filtered_scores.append(scores[i])
    return sorted(filtered_scores, key=itemgetter(0), reverse=True)


def find_first_minimum(series):
    """Returns the location of the first local minimum in a time series.

    Args:
        series (np.array): A one-dimensional array of time series data.

    Returns:
        integer: The index of the minimum in the array.

    """
    length = series.size

    smaller_right = np.zeros((length), dtype=np.bool)
    smaller_right[:-1] = np.less(series[:-1], series[1:])

    smaller_left = np.zeros((length), dtype=np.bool)
    smaller_left[1:] = np.less(series[1:], series[:-1])

    mins = np.logical_and(smaller_left, smaller_right)
    return np.argmax(mins)


def find_grid(hspace_angle, max_separation, graduations):
    """Returns the separation between graduations of the ruler.

    Args:
        hspace_angle: Bins outputted from :py:meth:`hough_transform`, but for only a single angle.
        max_separation: Maximum size of the *largest* graduation.
        graduations: List of graduation spacings in order of spacing size, normalised with respect to the
                     smallest graduation

    Returns:
        int: Separation between graduations in pixels

    """
    ratios = np.array(graduations) / (1.0 * graduations[0])

    autocorrelation = acf(hspace_angle, nlags=max_separation, unbiased=True)

    # size of autocorrelation output array can be less than max_separation if hspace_angle array is small
    max_separation = min(max_separation, autocorrelation.size - 1)

    min_size = find_first_minimum(autocorrelation)
    logging.info('Min size between graduations is {}'.format(min_size))

    separations = np.linspace(min_size, max_separation / graduations[-1], max_separation)

    uniform_separations = np.arange(max_separation + 1)
    offset_separations = separations[:, np.newaxis] * ratios[np.newaxis, :]
    score = np.sum(np.interp(offset_separations, uniform_separations, autocorrelation), axis=1)

    plt.plot(separations, score)
    plt.show()

    num_scores = ratios.size * 4
    best_scores = np.argsort(score)[-num_scores:]
    best_separation = list(zip(score[best_scores], separations[best_scores]))

    logging.info('Line separation candidates are:')
    for s in best_separation:
        logging.info(s)

    best_separation = remove_multiples(best_separation, ratios)

    logging.info('Line separation candidates after removing multiples:')
    for s in best_separation:
        logging.info(s)

    return best_separation[0][1]


def candidate_rulers(binary_image, n=10, output_images=False):
    """Return a list of potential locations for rulers in a binary image.

    Args:
        binary_image: 2D image, where 0 is off and 255 is on.
        n: Maximum number of candidate rulers to return. There may be *less* candidate rulers detected, but if
           there are more than n then only the best n will be used.
        output_images (optional): If True, will also return binary images representing the actual ruler
                                  (instead of just a bounding box).

    Returns:
        list: List of candidate :py:class:`Ruler` objects.

    """
    components = largest_components(binary_image, n)
    mean_area = np.mean([component.area for component in components])
    candidate_components = [c for c in components if c.area >= mean_area]
    rulers = [Ruler.from_box(c.bounding_box) for c in candidate_components]
    if output_images:
        return rulers, [c.draw(filled=True) for c in candidate_components]
    else:
        return rulers


def find_grid_from_ruler(ruler):
    """Returns the separation between graduations of the ruler, given a Ruler object.

    Wraps the :py:meth:`find_grid` method to use embedded values of a Ruler object instead.

    Args:
        ruler: A Ruler object. It must have its graduations defined.

    Returns:
        bool: True for success, False otherwise. Likely problem is the Ruler has not been defined adequately.

    """
    max_graduation_size = int(max(ruler.bounds.width, ruler.bounds.height))
    hspace_angle = ruler.hspace[:, ruler.angle_index]
    try:
        ruler.separation = find_grid(hspace_angle, max_graduation_size, ruler.graduations)
    except TypeError:
        return False
    return True


def resize_max(image, max_dim_size):
    """Scale image so that the maximum size in either dimension is the given value.

    Args:
        image: A three channel image.
        max_dim_size: After resizing, this will be the new maximum dimension value.

    Returns:
        ndarray: A three channel image of the same type as the input image.

    Note:
        This will only downscale images. Smaller images will be preserved at their current size.

    """
    max_image_dim = np.max(image.shape[:2])
    if max_image_dim > max_dim_size:
        scale_factor = max_dim_size / max_image_dim
        height, width = image.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        image = cv2.resize(image, (new_width, new_height))
    return image


def largest_contiguous_subarray(label_array):
    """Returns the largest contiguous subarray of identical values

    Args:
        label_array: An array of integers

    Returns:
        slice: A slice from label_array denoting the subarray

    """
    block_labels = np.cumsum(np.abs(np.diff(label_array)))
    blocks = find_objects(block_labels)
    block_sizes = [np.sum(label_array[b[0]]) for b in blocks]
    logging.info(block_sizes)
    return blocks[np.argmax(block_sizes)][0]


def cluster_series(series):
    """Cluster an array into two labels, based on the elements values.

    Args:
        series: A one-dimensional array of values

    Returns:
        array: An array of labels, where 0 is part of cluster 0, 1 is part of cluster 1 etc.

    """
    clustering = KMeans(n_clusters=2).fit(np.reshape(series, (-1, 1)))
    cluster_centres = clustering.cluster_centers_.flatten()
    labels = clustering.labels_

    labels2 = np.zeros_like(labels)
    cluster_centres_sorted = np.argsort(cluster_centres)
    for i, cluster_index in enumerate(cluster_centres_sorted):
        labels2[labels == cluster_index] = i
    logging.info(cluster_centres)
    logging.info(cluster_centres[cluster_centres_sorted])
    return labels2


def align_image_to_ruler(image, ruler):
    """Returns a rotated version of the image that is aligned with the major axis of a ruler.

    """
    angle = ruler.angles[ruler.angle_index]
    if abs(angle - np.pi / 2) < 0.1:
        rotated_image = np.transpose(image)[:, ::-1].copy()
    else:
        rotated_image = image
    return rotated_image


def crop_to_ruler(binary_image, ruler):
    """Returns a restricted crop of the input image that contains only the ruler.

    Note:
        This assumes that the ruler is aligned with the image axes

    Args:
        binary_image: 2D Binary image, where 0 is off and 255 is on.
        ruler: A Ruler object. Only the Hough space is used, and not the location.

    Returns:
        ndarray: An image of the same type as the input image, but smaller.

    """
    hspace_angle = ruler.hspace[:, ruler.angle_index]
    lines = np.interp(np.arange(binary_image.shape[1]), ruler.distances, hspace_angle)
    lines /= np.max(lines)

    component = largest_components(1 - binary_image, 1)[0]
    component_image = skeletonize(component.draw(filled=True, image=(1 - binary_image), color=0))
    line_image = component_image * np.tile(lines, (binary_image.shape[0], 1))

    rows = np.sum(line_image, axis=1)
    # remove outliers that distort clustering
    rows = np.clip(rows, 0, np.percentile(rows, 98))
    labels = cluster_series(rows)

    height, width = binary_image.shape[:2]
    crop = largest_contiguous_subarray(labels)
    crop_size = crop.stop - crop.start
    border_size = int(0.1 * crop_size)
    crop_border = slice(np.clip(crop.start - border_size, 0, height),
                        np.clip(crop.stop + border_size, 0, height))
    logging.info(crop)
    logging.info(crop_border)
    return binary_image[crop_border, :]


def restrict_search_space(binary_image, ruler):
    """Returns a restricted crop of the input image that contains only the ruler.

    A convenience method. Given the ruler parameters (angle and line distances), this crops the input image to
    include only the part that most likely contains the ruler, excluding as much excess as possible.

    Args:
        binary_image: 2D Binary image, where 0 is off and 255 is on.
        ruler: A Ruler object. Only the Hough space is used, and not the location.

    Returns:
        ndarray: An image of the same type as the input image, but smaller.

    """
    binary_image = align_image_to_ruler(binary_image, ruler)
    return crop_to_ruler(binary_image, ruler)


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
    original_height, original_width = image.shape[:2]
    image = resize_max(image, 2000)
    distance *= image.shape[0] / original_height

    height, width = image.shape[:2]
    image, mask = find_ruler(image)
    binary_image = mask * threshold(image, mask)

    if binary_image[mask].mean() > 128:
        binary_image[mask] = 255 - binary_image[mask]
    cv2.imwrite('binary_image.png', binary_image)
    remove_large_components(binary_image, max(height, width))
    cv2.imwrite('binary_image_reduced.png', binary_image)
    edges = find_edges(255 - binary_image)
    cv2.imwrite('edges.png', edges * 255)
    hspace, angles, distances = hough_transform(edges)
    features = hspace_features(hspace, splits=16)
    angle_index = best_angles(np.array(features))

    distance *= graduations[0]
    graduations = np.array(graduations) / graduations[0]

    max_graduation_size = int(max(image.shape))
    line_separation_pixels = find_grid(hspace[:, angle_index], max_graduation_size, graduations)

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return distance / line_separation_pixels
