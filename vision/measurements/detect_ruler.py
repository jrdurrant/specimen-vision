from skimage.transform import hough_line
from skimage.morphology import skeletonize
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf
from operator import itemgetter
from functools import total_ordering
from vision.segmentation.segmentation import largest_components

@total_ordering
class Ruler(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.indices = np.s_[y:(y + height), x:(x + width)]

        self.hspace = None
        self.score = 0
        self.angle_index = None

    def __lt__(self, other):
        return self.score < other.score
    def __eq__(self, other):
        return self.score == other.score

def threshold(image):
    threshold_val, binary_image = cv2.threshold(image[:, :, 1], 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def find_edges(binary_image):
    return skeletonize(1 - binary_image / 255)

def fill_gaps(edges, iterations=2):
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
    n = hspace_angle.shape[0]
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
    mean_freq = np.sum(values * freq) / np.sum(freq)
    return np.mean(np.power(values - mean_freq, 2) * freq)

def best_angle(hspace, distances):
    num_angles = hspace.shape[1]
    spread = np.zeros(num_angles)
    for i in range(num_angles):
        nz = np.nonzero(hspace[:, i])[0]
        spread[i] = var_freq(distances[nz[0]:nz[-1]], hspace[nz[0]:nz[-1], i])
    return np.argmax(spread), np.max(spread)

def candidate_rulers(binary_image, n=10):
    filled_image, boxes, areas = largest_components(binary_image, n, separate=True)
    bounding_boxes = (box for box, area in zip(boxes, areas) if area > np.mean(areas))
    return [Ruler(*box) for box in bounding_boxes]

def find_ruler(binary_image, num_candidates):
    rulers = candidate_rulers(binary_image, num_candidates)

    for ruler in rulers:
        binary_ruler_image = binary_image[ruler.indices]
        edges = fill_gaps(find_edges(binary_ruler_image))

        hspace, angles, distances = hough_transform(edges)
        angle_index, angle_score = best_angle(hspace, distances)
        ruler.score = angle_score
        ruler.angle_index = angle_index
        ruler.hspace = hspace

    return max(rulers)

logging.basicConfig(filename='ruler.log', filemode='w', level=logging.DEBUG, format='%(levelname)s %(message)s')
image = cv2.imread('BMNHE_500606.JPG')
height, width = image.shape[:2]
binary_image = threshold(image)

ruler = find_ruler(binary_image, num_candidates=10)

separation = find_grid(ruler.hspace[:, ruler.angle_index], int(image.shape[1] * 0.8), [1, 2, 20])
logging.info('Line separation: {:.3f}'.format(separation))