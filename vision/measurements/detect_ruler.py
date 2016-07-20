from skimage.transform import hough_line
from skimage.morphology import skeletonize
import cv2
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from operator import itemgetter
from vision.segmentation.segmentation import largest_components
from scipy.stats import entropy
from vision import Ruler
from sklearn.cluster import KMeans
from scipy.ndimage import find_objects
from skimage.filters import threshold_otsu


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


def hough_transform(binary_image):
    """Compute a Hough Transform on a binary image to detect straight lines

    Args:
        binary_image: 2D image, where 0 is off and 255 is on.

    Returns:
        (ndarray, array, array): Bins, angles, distances
                 Values of the bins after the Hough Transform, where the value at (i, j)
                 is the number of 'votes' for a straight line with distance[i] perpendicular to the origin
                 and at angle[j]. Also returns the corresponding array of angles and the corresponding array
                 of distances.

    """
    hspace, angles, distances = hough_line(binary_image, theta=np.linspace(0, np.pi, 180, endpoint=False))
    return hspace.astype(np.float32), angles, distances


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
            for r in ratios[1:]:
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
    min_size = find_first_minimum(autocorrelation)
    logging.info('Min size between graduations is {}'.format(min_size))

    separations = np.linspace(min_size, max_separation / graduations[-1], max_separation)

    uniform_separations = np.arange(max_separation + 1)
    offset_separations = separations[:, np.newaxis] * ratios[np.newaxis, :]
    score = np.sum(np.interp(offset_separations, uniform_separations, autocorrelation), axis=1)

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


def var_freq(values, freq):
    """Compute variance from values and their frequencies

    Args:
        values (array): 1-d array of values
        freq (array): Frequency of occurence in the data of the corresponding value
    Returns:
        float32: Variance of the data
    """
    mean_value = np.sum(values * freq) / (np.sum(freq))
    return np.mean(np.power(values - mean_value, 2) * freq)


def best_angle(features, feature_range):
    """Return the angle most likely to represent a ruler's graduation, given the bins resulting from a
    Hough Transform.

    Args:
        features: Feature vector describing the bins returned after the Hough Transform.
        feature_range: Range of the values of the features, given feature vectors for all candidate rulers.

    Returns:
        (int, float): The best angle index and its score, for the given features.

    Note:
        The returned angle index refers to an element in an angles array, and not the actual angle value.

    """
    num_features = len(features)
    for i in range(num_features):
        features[i] = (features[i] - feature_range[i][0]) / (feature_range[i][1] - feature_range[i][0])

    spread = features[0] - features[1]
    spread[(features[0] == 0) & (features[1] == 0)] = np.min(spread) - 1

    spread_global = np.zeros_like(spread)
    num_angles = spread.size
    weight = np.arange(num_angles) * np.arange(num_angles)[::-1]
    weight = weight.astype(np.float32) / np.max(weight)
    total_weight = np.sum(weight)
    for i in range(num_angles):
        current_weight = np.roll(weight, i)
        spread_global[i] = spread[i] - np.sum(spread * current_weight) / total_weight

    return np.argmax(spread_global), np.max(spread_global)


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


def get_hspace_features(hspace, distances):
    """Compute the features describing the Hough Transform bins.

    Args:
        hspace: Bins outputted from :py:meth:`hough_transform`.
        distances: Array of distances corresponding to the 2D hspace array.

    Returns:
        [float, float]: The variance and entropy of each angle, using all distances.

    """
    num_angles = hspace.shape[1]
    sample_variance = np.zeros(num_angles)
    sample_entropy = np.zeros(num_angles)
    for i in range(num_angles):
        nz = np.nonzero(hspace[:, i])[0]
        if nz.size > 1:
            sample_variance[i] = var_freq(distances[nz[0]:nz[-1]], hspace[nz[0]:nz[-1], i])
            sample_entropy[i] = entropy(hspace[nz[0]:nz[-1], i])
    return [sample_variance, sample_entropy]


def find_ruler(binary_image, num_candidates):
    """Return the location of a ruler in an image.

    Args:
        binary_image: 2D Binary image, where 0 is off and 255 is on.
        num_candidates: Number of candidate rulers to assess. This should be able to be low (<5), but higher
                        values allow for a more thorough search.

    Returns:
        Ruler: The most likely candidate :py:class:`Ruler`.

    """
    rulers, images = candidate_rulers(binary_image, num_candidates, output_images=True)

    for ruler, image in zip(rulers, images):
        binary_ruler_image = (image / 255.0) * binary_image[ruler.indices]
        edges = fill_gaps(find_edges(binary_ruler_image))
        hspace, angles, distances = hough_transform(edges)
        ruler.hspace = hspace
        ruler.angles = angles
        ruler.distances = distances

    features_ruler = [get_hspace_features(r.hspace, r.distances) for r in rulers]
    features_separate = list(zip(*features_ruler))
    features = [np.concatenate(feature) for feature in features_separate]
    feature_range = [(np.min(f), np.max(f)) for f in features]

    for i, ruler in enumerate(rulers):
        angle_index, angle_score = best_angle(features_ruler[i], feature_range)
        ruler.score = angle_score
        ruler.angle_index = angle_index
        logging.info('Ruler angle index is {}, score is {}'.format(angle_index, angle_score))

    best_ruler = 0
    for i, ruler in enumerate(rulers):
        if rulers[i] > rulers[best_ruler]:
            best_ruler = i
    logging.info('Best ruler angle index is {}, score is {}'.format(rulers[best_ruler].angle_index, rulers[best_ruler].score))

    return max(rulers)


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
    binary_image = threshold(image) / 255

    ruler = find_ruler(binary_image, num_candidates=10)
    binary_image_cropped = restrict_search_space(binary_image, ruler)
    ruler = find_ruler(binary_image_cropped, num_candidates=10)

    distance *= graduations[0]
    ruler.graduations = np.array(graduations) / graduations[0]

    find_grid_from_ruler(ruler)

    line_separation_pixels = ruler.separation

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return distance / line_separation_pixels
