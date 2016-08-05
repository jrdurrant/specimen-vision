import numpy as np
from skimage.feature import canny
from skimage.draw import set_color, polygon_perimeter
from vision.image_functions import threshold
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import find_contours, regionprops, label
from skimage.transform import SimilarityTransform
import csv
import cv2
import matplotlib.pyplot as plt
from skimage import draw
from sklearn.cluster import KMeans
import scipy


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 2:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


shapes = [read_shape(i) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=5)

mu, phi, sigma2 = shape_model

# for d in range(5):
#     for h_v in np.linspace(-2, 2, 10):
#         h = np.zeros((5, 1))
#         h[d] = h_v
#         s = mu + phi @ h
#         s = s.reshape(-1, 2)
#         plt.plot(s[:, 0], s[:, 1])
#     plt.show()

wings_image = get_test_image('wing_area', 'pinned', '0.png')
cv2.imwrite('wings.png', wings_image)
edges = canny(wings_image[:, :, 1], 2.5)
cv2.imwrite('wing_edge.png', 255 * edges)

saliency = saliency_dragonfly(wings_image)
distance = scipy.ndimage.distance_transform_edt(~edges)

kmeans = KMeans(n_clusters=8)
thresh = threshold(saliency)
indices_vector = np.array(np.where(thresh)).T
saliency_vector = saliency[thresh].reshape(-1, 1)
distance_vector = distance[thresh].reshape(-1, 1)
color_vector = wings_image[thresh].reshape(-1, 3)

distance2 = np.copy(distance)
distance2[~thresh] = 0
cv2.imwrite('distance.png', 255 * distance2 / distance2.max())
thresh2 = threshold(distance2)
output_image = (0.5 + 0.5 * thresh2)[:, :, np.newaxis] * wings_image
cv2.imwrite('distance2.png', output_image)
regions = regionprops(label(thresh2))
wings = [r for r in regions if r.filled_area > 1000]
initial_rotation = np.zeros(3)
initial_scale = np.zeros(3)
initial_translation = np.zeros((3, 2))
for i, wing in enumerate(wings):
    tform = SimilarityTransform(rotation=wing.orientation)
    major = wing.major_axis_length * 1.125
    minor = wing.minor_axis_length * 1.125
    initial_scale[i] = np.sqrt(np.power(major / 2, 2) + np.power(minor / 2, 2))
    initial_rotation[i] = wing.orientation
    initial_translation[i, :] = wing.centroid
    coords = np.array([[-(minor / 2), -(major / 2)], [-(minor / 2), (major / 2)], [(minor / 2), (major / 2)], [(minor / 2), -(major / 2)]])
    rotated_coords = tform(coords) + wing.centroid
    box_coords = polygon_perimeter(rotated_coords[:, 0], rotated_coords[:, 1])
    set_color(wings_image, box_coords, [0, 0, 255])
cv2.imwrite('distance_box.png', wings_image)

shape_model[0][::2] *= -1

inference = subspace_shape.infer(edges,
                                 *shape_model,
                                 scale_estimate=initial_scale[2],
                                 rotation=initial_rotation[2],
                                 translation=initial_translation[2, [1, 0]])
for iteration in range(100):
    fitted_shape = next(inference)

output_image = np.copy(wings_image)
points = fitted_shape[:, [1, 0]]
perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 255])
cv2.imwrite('wings_template.png', output_image)

# features = np.concatenate((0.05 * indices_vector, saliency_vector, 0.5 * distance_vector, 0.5 * color_vector), axis=1)
# # features = indices
# kmeans.fit(features)
# label = np.zeros_like(wings_image)

# subset = np.arange(kmeans.labels_.size)
# np.random.shuffle(subset)
# subset = subset[:5000]
# plt.scatter(features[subset, 1], features[subset, 0], c=kmeans.labels_[subset])
# plt.show()
