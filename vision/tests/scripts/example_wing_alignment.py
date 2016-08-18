import numpy as np
from skimage.feature import canny
from skimage.draw import set_color, polygon_perimeter
from vision.image_functions import threshold
from vision.segmentation.segment import saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import find_contours, regionprops, label
from skimage.transform import SimilarityTransform
import csv
import matplotlib.pyplot as plt
from skimage import draw
from sklearn.cluster import KMeans
import scipy
from operator import attrgetter
from skimage.morphology import skeletonize
from matplotlib import cm
from vision.io_functions import write_image
from skimage.color import rgb2lab
from skimage.segmentation import random_walker


def visualize_modes(shape_model):
    mu, phi, sigma2 = shape_model
    K = phi.shape[1]

    n = 10
    colors = [cm.bone(i) for i in np.linspace(0.35, 1, n)]
    for d in range(K):
        plt.gca().set_color_cycle(colors)
        for h_v in np.linspace(2, -2, n):
            h = np.zeros((K, 1))
            h[d] = h_v
            s = mu + phi @ h
            s = s.reshape(-1, 2)
            plt.plot(s[:, 0], s[:, 1])
        plt.savefig('mode{}.png'.format(d), transparent=True)
        plt.clf()


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 2:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


def smoothed_shape(shape, iterations=3):
    shape_smooth = shape
    for iteration in range(iterations):
        shape_smooth_old = shape_smooth
        shape_smooth = np.zeros((2 * (shape_smooth_old.shape[0] - 1), 2))
        shape_smooth[0, :] = shape_smooth_old[0, :]
        shape_smooth[-1, :] = shape_smooth_old[-1, :]
        for i in range(1, shape_smooth_old.shape[0] - 1):
            shape_smooth[2 * i - 1, :] = 0.75 * shape_smooth_old[i, :] + 0.25 * shape_smooth_old[i - 1, :]
            shape_smooth[2 * i, :] = 0.75 * shape_smooth_old[i, :] + 0.25 * shape_smooth_old[i + 1, :]
    return shape_smooth

shapes = [smoothed_shape(read_shape(i)) for i in range(4)]

wings_image = get_test_image('wing_area', 'cropped', 'unlabelled', '8.png')
# write_image('wings.png', wings_image)
edges = canny(wings_image[:, :, 1], 3)

saliency = saliency_dragonfly(wings_image)
thresh = threshold(saliency)

background = threshold(scipy.ndimage.distance_transform_edt(~thresh))

contours = find_contours(thresh, level=0.5)
outline = max(contours, key=attrgetter('size')).astype(np.int)
outline_image = np.zeros_like(edges)
draw.set_color(outline_image, (outline[:, 0], outline[:, 1]), True)

edges = skeletonize(edges)
gaps = scipy.ndimage.filters.convolve(1 * edges, np.ones((3, 3)), mode='constant', cval=False)
edges[(gaps == 2) & ~edges] = True
edges = skeletonize(edges)
# write_image('wing_edge.png', edges)

distance = scipy.ndimage.distance_transform_edt(~edges)

labels = label(edges)
num_labels = np.max(labels)
edge_distance = np.zeros(num_labels + 1)
for i in range(num_labels + 1):
    other_distance = scipy.ndimage.distance_transform_edt(~((labels > 0) & (labels != (i))))
    edge_distance[i] = np.median(other_distance[labels == (i)])

regions = regionprops(labels)

edge_lengths = np.zeros_like(labels)
for i, edge in enumerate(sorted(regions, key=attrgetter('filled_area'))):
    edge_lengths[labels == edge.label] = edge.filled_area

# write_image('labels.png', labels / labels.max())

scores = edges.shape[0] * np.exp(-edge_lengths**4 / (8 * edges.shape[0]**4))
# write_image('edges_wing.png', scores / scores.max())

kmeans = KMeans(n_clusters=8)
indices_vector = np.array(np.where(thresh)).T
saliency_vector = saliency[thresh].reshape(-1, 1)
distance_vector = distance[thresh].reshape(-1, 1)
color_vector = wings_image[thresh].reshape(-1, 3)

distance2 = np.copy(distance)
distance2[~thresh] = 0
# write_image('distance.png', distance2 / distance2.max())
thresh2 = threshold(distance2)
output_image = (0.5 + 0.5 * thresh2)[:, :, np.newaxis] * wings_image
# write_image('distance2.png', output_image)
wing_labels = label(thresh2)
regions = regionprops(wing_labels)
wings = sorted([r for r in regions if r.filled_area > 1000], key=attrgetter('filled_area'), reverse=True)

labels = np.zeros_like(wing_labels)
labels[background] = 1
for index, wing in enumerate(wings):
    labels[wing_labels == wing.label] = index + 2

seg = random_walker(rgb2lab(wings_image), labels, multichannel=True)
write_image('seg.png', seg / seg.max())

initial_rotation = np.zeros(3)
initial_scale = np.zeros(3)
initial_translation = np.zeros((3, 2))
for i, wing in enumerate(wings):
    tform = SimilarityTransform(rotation=wing.orientation)
    major = wing.major_axis_length * 1.125
    minor = wing.minor_axis_length * 1.125
    initial_scale[i] = np.sqrt(np.power(major / 2, 2) + np.power(minor / 2, 2))
    initial_rotation[i] = -wing.orientation
    initial_translation[i, :] = wing.centroid
    coords = np.array([[-(minor / 2), -(major / 2)],
                       [-(minor / 2),  (major / 2)],
                       [(minor / 2),  (major / 2)],
                       [(minor / 2), -(major / 2)]])
    rotated_coords = tform(coords) + wing.centroid
    box_coords = polygon_perimeter(rotated_coords[:, 0], rotated_coords[:, 1])
    set_color(wings_image, box_coords, [0, 0, 1])
# write_image('distance_box.png', wings_image)

aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=8)

# slices = [slice(13, -2)] + [slice(start, None) for start in range(13)[::-1]]
slices = [slice(None)]

inference = subspace_shape.infer(edges,
                                 edge_lengths,
                                 *shape_model,
                                 update_slice=slices[0],
                                 scale_estimate=initial_scale[1],
                                 rotation=initial_rotation[1],
                                 translation=initial_translation[1, [1, 0]])

inference.send(None)
for i, s in enumerate(slices):
    for iteration in range(100):
        fitted_shape, closest_edge_points = inference.send(s)

        output_image = 0.5 * (wings_image + edges[:, :, np.newaxis])

        points = closest_edge_points[:, [1, 0]]
        perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
        draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 1, 0])

        points = fitted_shape[:, [1, 0]]
        perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
        draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 1])
        if iteration % 20 == 0:
            write_image('wings_template_{}.png'.format(iteration), output_image)
