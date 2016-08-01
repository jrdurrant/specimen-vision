import numpy as np
from skimage.feature import canny
from skimage.draw import set_color
from vision.image_functions import threshold
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import find_contours
import csv
import cv2
import matplotlib.pyplot as plt


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 3:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


shapes = [read_shape(i) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=5)

image = get_test_image('wing_area', 'cropped', '4.png')
saliency_map = saliency_dragonfly(image)
cv2.imwrite('saliency.png', saliency_map)

contours = find_contours(threshold(saliency_map), 0.5)
wing_contour = max(contours, key=len).astype(np.int)

saliency_contour = np.zeros_like(image)
set_color(saliency_contour, (wing_contour[:, 0], wing_contour[:, 1]), [0, 0, 255])
cv2.imwrite('contour.png', saliency_contour)
contour = saliency_contour[:, :, 2] > 0

wings_image = image
cv2.imwrite('wings.png', wings_image)
edges = canny(wings_image[:, :, 1], 2.5)
edges = 0.5 * (contour + edges)
edges = threshold(edges)
cv2.imwrite('wing_edge.png', 255 * edges)

fitted_shape = subspace_shape.infer(edges, *shape_model)
fitted_shape[:, 1] = wings_image.shape[0] - fitted_shape[:, 1]
for vertex in fitted_shape:
    wings_image[int(vertex[1]), int(vertex[0]), :] = [0, 0, 255]
cv2.imwrite('wings_template.png', wings_image)

mu, phi, sigma2 = shape_model

for d in range(5):
    for h_v in np.linspace(-2, 2, 10):
        h = np.zeros((5, 1))
        h[d] = h_v
        s = mu + phi @ h
        s = s.reshape(-1, 2)
        plt.plot(s[:, 0], s[:, 1])
    plt.show()
