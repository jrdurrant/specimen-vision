import numpy as np
from skimage.feature import canny
from sklearn.neighbors import NearestNeighbors
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
import menpo
import menpofit
from operator import attrgetter
from skimage.morphology import closing, disk, skeletonize
from matplotlib import cm


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


shapes = [read_shape(i) for i in range(4)]

wings_image = get_test_image('wing_area', 'cropped', 'unlabelled', '8.png')
cv2.imwrite('wings.png', wings_image)
edges = canny(wings_image[:, :, 1], 3)

saliency = saliency_dragonfly(wings_image)
thresh = threshold(saliency)

contours = find_contours(thresh, level=0.5)
outline = max(contours, key=attrgetter('size')).astype(np.int)
outline_image = np.zeros_like(edges)
draw.set_color(outline_image, (outline[:, 0], outline[:, 1]), True)

edges = skeletonize(edges)
gaps = scipy.ndimage.filters.convolve(1 * edges, np.ones((3, 3)), mode='constant', cval=False)
edges[(gaps == 2) & ~edges] = True
edges = skeletonize(edges)
cv2.imwrite('wing_edge.png', 255 * edges)

distance = scipy.ndimage.distance_transform_edt(~edges)

labels = label(edges)
num_labels = np.max(labels)
# other_distance = [scipy.ndimage.distance_transform_edt(~((labels > 0) & (labels != l))) for l in range(1, num_labels + 1)]
edge_distance = np.zeros(num_labels + 1)
for i in range(num_labels + 1):
    other_distance = scipy.ndimage.distance_transform_edt(~((labels > 0) & (labels != (i))))
    edge_distance[i] = np.median(other_distance[labels == (i)])

regions = regionprops(labels)

edge_lengths = np.zeros_like(labels)
for i, edge in enumerate(sorted(regions, key=attrgetter('filled_area'))):
    edge_lengths[labels == edge.label] = edge.filled_area

cv2.imwrite('labels.png', labels * 255 / labels.max())

scores = edges.shape[0] * np.exp(-edge_lengths**4 / (8 * edges.shape[0]**4))
cv2.imwrite('edges_wing.png', scores * 255 / scores.max())

kmeans = KMeans(n_clusters=8)
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
wings = sorted([r for r in regions if r.filled_area > 1000], key=attrgetter('filled_area'), reverse=True)
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
                       [ (minor / 2),  (major / 2)],
                       [ (minor / 2), -(major / 2)]])
    rotated_coords = tform(coords) + wing.centroid
    box_coords = polygon_perimeter(rotated_coords[:, 0], rotated_coords[:, 1])
    set_color(wings_image, box_coords, [0, 0, 255])
cv2.imwrite('distance_box.png', wings_image)

aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=8)

visualize_modes(shape_model)

# slices = [slice(13, -2)] + [slice(start, None) for start in range(13)[::-1]]
slices = [slice(None)]

inference = subspace_shape.infer(edges,
                                 edge_lengths,
                                 *shape_model,
                                 update_slice=slices[0],
                                 scale_estimate=initial_scale[0],
                                 rotation=initial_rotation[0],
                                 translation=initial_translation[0, [1, 0]])

inference.send(None)
for i, s in enumerate(slices):
    for iteration in range(100):
        fitted_shape, closest_edge_points = inference.send(s)

        output_image = 0.5 * (wings_image + 255 * edges[:, :, np.newaxis])

        points = closest_edge_points[:, [1, 0]]
        perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
        draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 255, 0])

        points = fitted_shape[:, [1, 0]]
        perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
        draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 255])
        cv2.imwrite('wings_template_{}.png'.format(iteration), output_image)

# training_images = menpo.io.import_images('/home/james/vision/vision/tests/test_data/wing_area/cropped/',
#                                          verbose=True)


# patch_aam = menpofit.aam.PatchAAM(training_images, group='PTS', patch_shape=(35, 35),
#                                   holistic_features=menpo.feature.fast_dsift,
#                                   verbose=True)

# fitter = menpofit.aam.LucasKanadeAAMFitter(patch_aam, n_shape=None, n_appearance=None)

# image = menpo.image.Image(np.transpose(wings_image[:, :, [2, 1, 0]], (2, 0, 1)))
# result = fitter.fit_from_shape(image, menpo.shape.PointCloud(fitted_shape[:, [1, 0]]))

# result.view(render_initial_shape=True, figure_size=(20, 20)).save_figure('fig.png', overwrite=True)
# result.view_iterations(figure_size=(20, 20)).save_figure('fig_iter.png', overwrite=True)

# features = np.concatenate((0.05 * indices_vector, saliency_vector, 0.5 * distance_vector, 0.5 * color_vector), axis=1)
# # features = indices
# kmeans.fit(features)
# label = np.zeros_like(wings_image)

# subset = np.arange(kmeans.labels_.size)
# np.random.shuffle(subset)
# subset = subset[:5000]
# plt.scatter(features[subset, 1], features[subset, 0], c=kmeans.labels_[subset])
# plt.show()
