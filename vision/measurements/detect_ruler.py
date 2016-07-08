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


@total_ordering
class Ruler(object):
    """A ruler from an image

    This class describes the rulers position in an image, as well as defining quantities that assess its
    suitability to represent an actual ruler and the parameters of that ruler

    Attributes:
        bounds: A Box object describing the bounding box of the ruler in its image
        indices: A NumPy slice object that can be used for cropping the ruler from its image
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

        self.indices = np.s_[y:(y + height), x:(x + width)]

        self.hspace = None
        self.angles = None
        self.distances = None

        self.score = None
        self.angle_index = None

        self.graduations = []
        self.separation = 0

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
            if abs(scores[i][1] / scores[j][1] % 1) < threshold:
                multiple = True
        if not multiple:
            filtered_scores.append(scores[i])
    return sorted(filtered_scores, key=itemgetter(0), reverse=True)


def find_grid(hspace_angle, nlags, graduations, min_size=4):
    ratios = np.array(graduations) / (1.0 * graduations[0])

    autocorrelation = acf(hspace_angle, nlags=nlags)
    separation = np.linspace(min_size, nlags / graduations[-1], nlags)
    score = np.zeros(nlags)
    for i, x in enumerate(separation):
        score[i] = np.sum(np.interp(x * ratios, range(nlags + 1), autocorrelation))

    num_scores = ratios.size * 4
    best_scores = np.argsort(score)[-num_scores:]
    best_separation = zip(score[best_scores], separation[best_scores])

    logging.info('Line separation candidates are:')
    for s in best_separation:
        logging.info(s)

    best_separation = remove_multiples(best_separation)

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


def best_angle(features, normalise_features=False, feature_range=None):
    num_features = len(features)
    if normalise_features:
        for i in range(num_features):
            features[i] = (features[i] - feature_range[i][0]) / (feature_range[i][1] - feature_range[i][0])

    spread = features[0] - features[1]
    spread[(features[0] == 0) & (features[1] == 0)] = np.min(spread) - 1
    return np.argmax(spread), np.max(spread)


def candidate_rulers(binary_image, n=10, output_images=False):
    filled_images, boxes, areas = largest_components(binary_image, n, separate=True)
    bounding_boxes = (box for box, area in zip(boxes, areas) if area > np.mean(areas))
    rulers = [Ruler(*box) for box in bounding_boxes]
    if output_images:
        return rulers, filled_images
    else:
        return rulers


def get_hspace_features(hspace, distances):
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
        angle_index, angle_score = best_angle(features_ruler[i],
                                              normalise_features=True,
                                              feature_range=feature_range)
        ruler.score = angle_score
        ruler.angle_index = angle_index
        logging.info('Ruler angle index is {}, score is {}'.format(angle_index, angle_score))

    return max(rulers)


def find_grid_from_ruler(ruler):
    max_graduation_size = int(max(ruler.bounds.width, ruler.bounds.height))
    hspace_angle = ruler.hspace[:, ruler.angle_index]
    ruler.separation = find_grid(hspace_angle, max_graduation_size, ruler.graduations)


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
    ruler.graduations = graduations

    find_grid_from_ruler(ruler)

    line_separation_pixels = ruler.separation

    logging.info('Line separation: {:.3f}'.format(line_separation_pixels))
    return distance * graduations[0] / line_separation_pixels
