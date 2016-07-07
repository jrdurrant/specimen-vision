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

import warnings
warnings.simplefilter('error')

logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


@total_ordering
class Ruler(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

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


def normalise(array):
    zero_mean_array = array - np.mean(array)
    return zero_mean_array / np.std(zero_mean_array)


def threshold(image):
    """Convert a full color image to a binary image

    Args:
        image (ndarray): BGR image of shape n x m x 3.

    Returns:
        ndarray: Binary image of shape n x m, where 0 is off and 255 is on.

    """
    threshold_val, binary_image = cv2.threshold(image[:, :, 1], 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def find_edges(binary_image):
    return skeletonize(1 - binary_image / 255)


def fill_gaps(edges, iterations=1):
    edges = edges * 1.0
    kernel = np.ones((3, 3))
    for i in range(iterations):
        edges = cv2.dilate(edges, kernel)
        edges = skeletonize(edges) * 1.0
    return edges * 255.0


def hough_transform(binary_image):
    hspace, angles, distances = hough_line(binary_image, theta=np.linspace(0, np.pi, 180, endpoint=False))
    return hspace.astype(np.float32), angles, distances


def remove_multiples(scores, ratios):
    n = len(scores)
    n_ratios = len(ratios)

    scores.sort(key=itemgetter(1))

    if n_ratios == 1:
        return scores
    else:
        filtered_scores = []
        for i in range(n):
            multiple = False
            for j in range(i):
                for k in range(1, n_ratios):
                    if np.power(scores[i][1] / scores[j][1] - ratios[k], 2) < 0.1:
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

    best_scores = np.argsort(score)[-10:]
    best_separation = zip(score[best_scores], separation[best_scores])

    logging.info('Line separation candidates are:')
    for s in best_separation:
        logging.info(s)

    best_separation = remove_multiples(best_separation, ratios)

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

    for i, (ruler, image) in enumerate(zip(rulers, images)):
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
    max_graduation_size = int(max(ruler.width, ruler.height))
    hspace_angle = ruler.hspace[:, ruler.angle_index]
    ruler.separation = find_grid(hspace_angle, max_graduation_size, ruler.graduations)


def ruler_line_separation(image):
    height, width = image.shape[:2]
    binary_image = threshold(image)

    ruler = find_ruler(binary_image, num_candidates=10)
    ruler.graduations = [0.5, 1, 10]

    find_grid_from_ruler(ruler)
    logging.info('Line separation: {:.3f}'.format(ruler.separation))
    return ruler.separation
