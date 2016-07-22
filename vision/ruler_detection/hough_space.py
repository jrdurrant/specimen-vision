import itertools
import numpy as np
from scipy.stats import entropy
from skimage.transform import hough_line


def average_local_entropy(arr, window_size=2):
    """Calculate the average entropy computed with a sliding window.

    Note:
        Assumes all elements of array are positive

    """
    if np.min(arr) < 0:
        raise ValueError("all elements of array must be posiive")
    arr = 1.0 * arr / np.sum(arr)
    log_arr = np.log(arr)
    log_arr = np.where(np.isinf(log_arr), 0, log_arr)

    kernel = np.ones(2 * window_size + 1)
    local_entropy_unnormalized = np.convolve(-arr * log_arr, kernel, mode='valid')
    local_sum = np.convolve(arr, kernel, mode='valid')
    local_entropy = (local_entropy_unnormalized / local_sum) + np.log(local_sum)

    local_entropy = np.where(local_sum == 0, 0, local_entropy)

    return np.sum(local_entropy) / arr.size


def hspace_angle_features(distance_bins, distances):
    non_zero_indices = np.nonzero(distance_bins)[0]
    if non_zero_indices.size > 10:
        non_zero_distance_bins = distance_bins[non_zero_indices[0]:non_zero_indices[-1]]
        non_zero_distances = distances[non_zero_indices[0]:non_zero_indices[-1]]
        return (var_freq(non_zero_distances, non_zero_distance_bins),
                average_local_entropy(non_zero_distance_bins))
    else:
        return np.nan, np.nan


def hspace_angle_scale(distance_bins, distances, splits=2):
    if splits <= 1:
        return hspace_angle_features(distance_bins, distances)
    else:
        split_arrays = zip(np.array_split(distance_bins, splits), np.array_split(distances, splits))
        downscaled_features = (hspace_angle_scale(bins_split, distances_split, splits / 2)
                               for bins_split, distances_split
                               in split_arrays)
        current_level_features = hspace_angle_features(distance_bins, distances)
        return tuple(itertools.chain(*downscaled_features)) + current_level_features


def hspace_features(hspace, distances, splits=2):
    num_angles = hspace.shape[1]
    return [hspace_angle_scale(hspace[:, i], distances, splits) for i in range(num_angles)]


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


def grid_hough_space(binary_image, grid=8):
    hough_spaces = [[[] for i in range(grid)] for i in range(grid)]

    height, width = binary_image.shape
    grid_height = np.ceil(height / grid)
    grid_width = np.ceil(width / grid)

    for i in range(grid):
        for j in range(grid):
            grid_i = slice(i * grid_height, (i + 1) * grid_height)
            grid_j = slice(j * grid_width, (j + 1) * grid_width)
            hough_space = hough_transform(binary_image[grid_i, grid_j])
            features = np.array(hspace_features(hough_space[0], hough_space[2], splits=4))
            hough_spaces[i][j] = features

    return hough_spaces
