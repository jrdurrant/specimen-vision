import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from collections import namedtuple

Color = namedtuple('Color', ('RGB', 'proportion'))
Segment = namedtuple('Segment', ('name', 'mask', 'num_colors'))

def dominant_colors(image, num_colors, mask=None):
    image = cv2.cvtColor(image / 255.0, cv2.cv.CV_BGR2Lab)

    if mask is not None:
        data = image[mask > 250]
    else:
        data = np.reshape(image, (-1, 3))

    # kmeans algorithm has inherent randomness - result will not be exactly the same 
    # every time. Fairly consistent with >= 30 iterations
    centroids, labels = kmeans2(data, num_colors, iter=30)
    counts = np.histogram(labels, bins=range(0, num_colors + 1), normed=True)[0]

    centroids_RGB = cv2.cvtColor(np.reshape(centroids, (-1, 1, 3)), cv2.cv.CV_Lab2BGR)[:, 0, :] * 255.0
    colors = [Color(centroid, count) for centroid, count in zip(centroids_RGB, counts)]
    colors.sort(key=lambda color: np.mean(color.RGB))

    return colors

def visualise_colors(colors, output_width, output_height):
    output = np.zeros((100, output_width, 3), dtype='float32')
    left = 0
    for color in dc:
        right = left + int(color.proportion * output_width)
        output[:, left:right, :] = color.RGB
        left = right

    output[:, right:output_width, :] = colors[-1].RGB

    return output