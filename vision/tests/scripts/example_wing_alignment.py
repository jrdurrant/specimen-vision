import numpy as np
from skimage.feature import canny
from skimage.draw import set_color
from vision.image_functions import threshold
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import find_contours, compare_ssim
from skimage.transform import SimilarityTransform, warp
import csv
import cv2
import matplotlib.pyplot as plt


def find_closest_points(edge_points, angles, shapes, images, n=5):
    num_points = edge_points.shape[0]
    num_images = len(images)

    closest_points = -np.ones((num_points, 2))

    for p in range(num_points):
        print('Point {}'.format(p))
        current_angle = angles[p]
        current_location = edge_points[p, :]

        coords = np.stack((np.linspace(-100, 100, n), np.zeros(n)), axis=1)
        tform = SimilarityTransform(rotation=(current_angle))
        aligned_coords = (tform(coords) + current_location).astype(np.int)

        similarity = np.zeros((n, num_images))

        template_patches = [extract_rotated_patch(image, shape[p, [1, 0]], current_angle, (32, 32))
                            for image_index, (image, shape)
                            in enumerate(zip(images, shapes))]

        for index, (y, x) in enumerate(aligned_coords):
            new_patch = extract_rotated_patch(images[0], [x, y], current_angle, (32, 32))
            for template_index, template_patch in enumerate(template_patches):
                similarity[index, template_index] = compare_ssim(template_patch[:, :, 1], new_patch[:, :, 1])
        best_offset_index = np.unravel_index(np.argmax(similarity), (n, num_images))[0]
        print('Best image is {}'.format(np.unravel_index(np.argmax(similarity), (n, num_images))[1]))
        closest_points[p, :] = aligned_coords[best_offset_index, :]
    return closest_points


def extract_rotated_patch(image, location, rotation, patch_size):
    """where rotation is rotation in radians clockwise from north
    """
    shift_y, shift_x = location
    tf_rotate = SimilarityTransform(rotation=(-rotation))
    tf_shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = SimilarityTransform(translation=[patch_size[0], patch_size[1]])
    tform = (tf_shift + (tf_rotate + tf_shift_inv))
    rotated = warp(image, tform.inverse, output_shape=(2 * patch_size[0] + 1, 2 * patch_size[1] + 1))
    return rotated * 255


def patch_distance(patch_a, patch_b):
    Gy_A, Gx_A, Gz_A = np.gradient(patch_a)
    Gy_B, Gx_B, Gz_B = np.gradient(patch_b)

    return np.sum(np.power(Gy_B[:, :, 0] - Gy_A[:, :, 0], 2) + np.power(Gx_B[:, :, 1] - Gx_A[:, :, 1], 2))


def local_angle(shape):
    n = shape.shape[0]
    offset = shape[np.mod(np.arange(0, n) + 1, n), :] - shape[np.mod(np.arange(0, n) - 1, n), :]
    return np.pi - np.arctan2(offset[:, 0], offset[:, 1])


def render_oriented_patches(image, shape, patch_size):
    angles = local_angle(shape)
    n = shape.shape[0]
    patch_height, patch_width = np.array(patch_size) * 2 + 1
    output_image = np.zeros((patch_height, n * patch_width, 3))
    for i in range(n):
        output_image[:, (i * patch_width):((i + 1) * patch_width), :] = extract_rotated_patch(image, shape[i, [1, 0]], angles[i] - np.pi / 2, patch_size)
    return output_image * 255


def visualize_modes(shape_model):
    mu, phi, sigma2 = shape_model

    for d in range(5):
        for h_v in np.linspace(-2, 2, 10):
            h = np.zeros((5, 1))
            h[d] = h_v
            s = mu + phi @ h
            s = s.reshape(-1, 2)
            plt.plot(s[:, 0], s[:, 1])
        plt.show()


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 2:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


def visualize_patches(image, shape, size=32):
    output_image = np.zeros_like(image)
    for x, y in shape:
        slice_y = slice(y - size, y + size + 1)
        slice_x = slice(x - size, x + size + 1)
        output_image[slice_y, slice_x, :] = image[slice_y, slice_x, :]
    return output_image


shapes = [read_shape(i) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=5)

images = [get_test_image('wing_area', 'cropped', '{}.png'.format(i)) for i in range(4)]

for i in range(4):
    cv2.imwrite('patches_{}.png'.format(i), render_oriented_patches(images[i], shapes[i], (32, 32)))

close = find_closest_points(shapes[0], local_angle(shapes[0]), shapes, images)

# saliency_map = saliency_dragonfly(image)
# cv2.imwrite('saliency.png', saliency_map)

# contours = find_contours(threshold(saliency_map), 0.5)
# wing_contour = max(contours, key=len).astype(np.int)

# saliency_contour = np.zeros_like(image)
# set_color(saliency_contour, (wing_contour[:, 0], wing_contour[:, 1]), [0, 0, 255])
# cv2.imwrite('contour.png', saliency_contour)
# contour = saliency_contour[:, :, 2] > 0

# wings_image = image
# cv2.imwrite('wings.png', wings_image)
# edges = canny(wings_image[:, :, 1], 2.5)
# edges = 0.5 * (contour + edges)
# edges = threshold(edges)
# cv2.imwrite('wing_edge.png', 255 * edges)

fitted_shape = subspace_shape.infer(edges, *shape_model)
fitted_shape[:, 1] = wings_image.shape[0] - fitted_shape[:, 1]
for vertex in fitted_shape:
    wings_image[int(vertex[1]), int(vertex[0]), :] = [0, 0, 255]
cv2.imwrite('wings_template.png', wings_image)
