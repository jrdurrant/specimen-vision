import numpy as np
from skimage.feature import canny
from skimage.draw import set_color, polygon_perimeter
from vision.image_functions import threshold
from vision.segmentation.segment import saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import regionprops, label
from skimage.transform import SimilarityTransform
import csv
import matplotlib.pyplot as plt
from skimage import draw
import scipy
from operator import attrgetter
from skimage.morphology import skeletonize
from matplotlib import cm
from vision.io_functions import write_image


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


def visualize_result(image, edge_image, shape, closest_points):
    output_image = 0.5 * (image + edge_image[:, :, np.newaxis])

    points = closest_points[:, [1, 0]]
    perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
    draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 1, 0])

    points = shape[:, [1, 0]]
    perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
    draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 1])
    return output_image

shapes = [smoothed_shape(read_shape(i)) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)
shape_model = subspace_shape.learn(aligned_shapes, K=8)

# wings_image = get_test_image('wing_area', 'cropped', 'unlabelled', '7.png')
wings_image = get_test_image('wing_area', 'pinned', '1.png')
edges = canny(wings_image[:, :, 1], 3)

saliency = saliency_dragonfly(wings_image)
thresh = threshold(saliency)

edges = skeletonize(edges)
gaps = scipy.ndimage.filters.convolve(1 * edges, np.ones((3, 3)), mode='constant', cval=False)
edges[(gaps == 2) & ~edges] = True
edges = skeletonize(edges)

distance = scipy.ndimage.distance_transform_edt(~edges)

labels = label(edges)

regions = regionprops(labels)

edge_lengths = np.zeros_like(labels)
for i, edge in enumerate(sorted(regions, key=attrgetter('filled_area'))):
    edge_lengths[labels == edge.label] = edge.filled_area

edges = edge_lengths > 500

distance2 = np.copy(distance)
distance2[~thresh] = 0
thresh2 = threshold(distance2)
wing_labels = label(thresh2)
regions = regionprops(wing_labels)
wings = sorted([r for r in regions if r.filled_area > 1000], key=attrgetter('filled_area'), reverse=True)

initial_rotation = np.zeros(3)
initial_scale = np.zeros(3)
initial_translation = np.zeros((3, 2))
for i, wing in enumerate(wings):
    tform = SimilarityTransform(rotation=wing.orientation)
    major = wing.major_axis_length * 1.125
    minor = wing.minor_axis_length * 1.125
    initial_scale[i] = 2 * np.sqrt(np.power(major / 2, 2) + np.power(minor / 2, 2))
    initial_rotation[i] = -wing.orientation
    initial_translation[i, :] = wing.centroid
    coords = np.array([[-(minor / 2), -(major / 2)],
                       [-(minor / 2),  (major / 2)],
                       [(minor / 2),  (major / 2)],
                       [(minor / 2), -(major / 2)]])
    rotated_coords = tform(coords) + wing.centroid
    box_coords = polygon_perimeter(rotated_coords[:, 0], rotated_coords[:, 1])
    set_color(wings_image, box_coords, [0, 0, 1])

slices = [slice(13, -2)] + [slice(start, None) for start in range(13)[::-1]]

for wing_index in range(len(wings)):
    inference = subspace_shape.infer(edges,
                                     edge_lengths,
                                     *shape_model,
                                     update_slice=slices[0],
                                     scale_estimate=initial_scale[wing_index],
                                     rotation=initial_rotation[wing_index],
                                     translation=initial_translation[wing_index, [1, 0]])

    inference.send(None)
    for i, s in enumerate(slices):
        for iteration in range(100):
            fitted_shape, closest_edge_points, h, psi = inference.send(s)

            # if iteration % 50 == 0:
            #     output_image = visualize_result(wings_image, edges, fitted_shape, closest_edge_points)
            #     write_image('wing_{}_slice_{}_iteration_{}.png'.format(wing_index, i, iteration), output_image)

    print(subspace_shape.similarity(edges, *shape_model, h, psi))
