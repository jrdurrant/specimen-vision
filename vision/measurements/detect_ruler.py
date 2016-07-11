from skimage.transform import hough_line
from skimage.morphology import skeletonize
import cv2
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from operator import itemgetter
from functools import total_ordering
from vision.segmentation.segmentation import largest_components
from scipy.stats import entropy

logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


class Box(object):
    """A 2D bounding box

    Attributes:
        x: x-coordinate of top-left corner
        y: y-coordinate of top-left corner
        width: width of bounding box
        height: height of bounding box

    """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    @property
    def indices(self):
        """(slice, slice): indices of the box for slicing from a larger array"""
        return slice(self.y, (self.y + self.height)), slice(self.x, (self.x + self.width))


@total_ordering
class Ruler(object):
    """A ruler from an image

    This class describes the rulers position in an image, as well as defining quantities that assess its
    suitability to represent an actual ruler and the parameters of that ruler

    Attributes:
        bounds: A Box object describing the bounding box of the ruler in its image
        hspace: An array of the bins resulting from the Hough transform on the cropped image of the ruler
        angles: An array of the angle bins used for the Hough transform :py:attr:`hspace`
        distances: An array of the distance bins used for the Hough transform :py:attr:`hspace`
        score: Measure of how well this ruler fits the expected pattern of a ruler
        angle_index: Index corresponding to the :py:attr:`angles`
            array of the angle of the graduations in the image
        graduations: List of the size of the gaps between different sized graduations, in ascending order
        separation: Distance in *pixels* between the smallest graduations

    """
    def __init__(self, x, y, width, height):
        self.bounds = Box(x, y, width, height)

        self.hspace = None
        self.angles = None
        self.distances = None

        self.score = None
        self.angle_index = None

        self.graduations = []
        self.separation = 0

    @property
    def indices(self):
        """(slice, slice): indices of the ruler in the image"""
        return self.bounds.indices

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score


def threshold(image):
    """Convert a full color image to a binary image

    Args:
        image (ndarray): BGR image of shape n x m x 3.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    threshold_val, binary_image = cv2.threshold(image[:, :, 1], 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


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


def remove_multiples(scores, threshold=0.05):
    """Returns a list where no elements are integer multiples of any other elements

    """
    n = len(scores)

    scores.sort(key=itemgetter(1))

    filtered_scores = []
    for i in range(n):
        multiple = False
        for j in range(i):
            ratio = scores[i][1] / scores[j][1]
            remainder = abs(ratio % 1)
            if remainder > 0.5:
                remainder -= 1
            if ratio > (1 + 2 * threshold) and remainder < threshold:
                multiple = True
        if not multiple:
            filtered_scores.append(scores[i])
    return sorted(filtered_scores, key=itemgetter(0), reverse=True)


def find_grid(hspace_angle, max_separation, graduations, min_size=4):
    """Returns the separation between graduations of the ruler.

    Args:
        hspace_angle: Bins outputted from :py:meth:`hough_transform`, but for only a single angle.
        max_separation: Maximum size of the *largest* graduation.
        graduations: List of graduation spacings in order of spacing size, normalised with respect to the
                     smallest graduation
        min_size: Minimum size of the *smallest* graduation, to avoid degenerate cases.

    Returns:
        int: Separation between graduations in pixels

    """
    ratios = np.array(graduations) / (1.0 * graduations[0])

    autocorrelation = acf(hspace_angle, nlags=max_separation)
    separation = np.linspace(min_size, max_separation / graduations[-1], max_separation)
    score = np.zeros(max_separation)
    for i, x in enumerate(separation):
        score[i] = np.sum(np.interp(x * ratios, range(max_separation + 1), autocorrelation))

    num_scores = ratios.size * 4
    best_scores = np.argsort(score)[-num_scores:]
    best_separation = zip(score[best_scores], separation[best_scores])

    logging.info('Line separation candidates are:')
    for s in best_separation:
        logging.info(s)

    best_separation = remove_multiples(best_separation)

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
    filled_images, boxes, areas = largest_components(binary_image, n, separate=True)
    bounding_boxes = (box for box, area in zip(boxes, areas) if area > np.mean(areas))
    rulers = [Ruler(*box) for box in bounding_boxes]
    if output_images:
        return rulers, filled_images
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
    features_separate = zip(*features_ruler)
    features = [np.concatenate(feature) for feature in features_separate]
    feature_range = [(np.min(f), np.max(f)) for f in features]

    for i, ruler in enumerate(rulers):
        angle_index, angle_score = best_angle(features_ruler[i], feature_range)
        ruler.score = angle_score
        ruler.angle_index = angle_index
        logging.info('Ruler angle index is {}, score is {}'.format(angle_index, angle_score))

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
    binary_image = threshold(image)

    ruler = find_ruler(binary_image, num_candidates=10)
    ruler.graduations = np.array(graduations) / graduations[0]

    find_grid_from_ruler(ruler)

    line_separation_pixels = ruler.separation

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return distance * graduations[0] / line_separation_pixels
