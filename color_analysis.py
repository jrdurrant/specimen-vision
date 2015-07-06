import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2, ClusterError
from collections import namedtuple

Color = namedtuple('Color', ('RGB', 'proportion'))

def dominant_colors(image, num_colors, mask=None):
    image = cv2.cvtColor(image / 255.0, cv2.cv.CV_BGR2Lab)

    if mask is not None:
        i, j = np.where(mask > 250)
        data = image[i, j, :]
    else:
        height, width = image.shape[:2]
        data = np.reshape(image, (height * width, 3))

    # kmeans algorithm has inherent randomness - result will not be exactly the same 
    # every time. Fairly consistent with >= 30 iterations
    centroids, labels = kmeans2(data, num_colors, iter=30)
    counts = np.histogram(labels, bins=range(0, num_colors + 1), normed=True)[0]

    centroids_RGB = cv2.cvtColor(np.reshape(centroids, (centroids.shape[0], 1, 3)), cv2.cv.CV_Lab2BGR)[:, 0, :] * 255.0
    colors = [Color(centroid, count) for centroid, count in zip(centroids_RGB, counts)]
    colors.sort(key=lambda color: np.mean(color.RGB))

    return colors

if __name__ == '__main__':
    input_folder = 'data/moths_wings/'
    filename = 'Basiothia_charis_f_MCB002_48.5_r.jpg'

    abdomen = cv2.imread(os.path.join(input_folder, 'abdomen_' + filename))
    color = cv2.imread(os.path.join(input_folder, 'color_' + filename))
    left_wing_front = cv2.imread(os.path.join(input_folder, 'left_wing_front_' + filename))
    left_wing_back = cv2.imread(os.path.join(input_folder, 'left_wing_back_' + filename))
    right_wing_front = cv2.imread(os.path.join(input_folder, 'right_wing_front_' + filename))
    right_wing_back = cv2.imread(os.path.join(input_folder, 'right_wing_back_' + filename))

    colors = np.concatenate((color, color), axis=0)
    mask = np.concatenate((left_wing_front, right_wing_front), axis=0)[:, :, 0]

    num_colors = 5

    dc = dominant_colors(colors.astype('float32'), num_colors, mask=mask)

    output_width = 100 * num_colors
    output = np.zeros((100, output_width, 3), dtype='float32')
    left = 0
    for color in dc:
        right = left + int(color.proportion * output_width)
        output[:, left:right, :] = color.RGB
        left = right

    cv2.imwrite('colors.png', output)