import numpy as np
from skimage.feature import canny
from skimage.draw import set_color
from vision.image_functions import threshold
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
from skimage.measure import find_contours
from skimage.transform import SimilarityTransform, warp
import csv
import cv2
import matplotlib.pyplot as plt
from skimage import draw


def patch_distance(patch_a, patch_b):
    Gy_A, Gx_A, Gz_A = np.gradient(patch_a)
    Gy_B, Gx_B, Gz_B = np.gradient(patch_b)

    return np.sum(np.power(Gy_B[:, :, 0] - Gy_A[:, :, 0], 2) + np.power(Gx_B[:, :, 1] - Gx_A[:, :, 1], 2))


def render_oriented_patches(image, shape, patch_size):
    angles = subspace_shape.local_angle(shape)
    n = shape.shape[0]
    patch_height, patch_width = np.array(patch_size) * 2 + 1
    output_image = np.zeros((patch_height, n * patch_width, 3))
    for i in range(n):
        output_image[:, (i * patch_width):((i + 1) * patch_width), :] = subspace_shape.extract_rotated_patch(image, shape[i, [1, 0]], angles[i] - np.pi / 2, patch_size)
    return output_image


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


def visualize_fitting(image, fitted_shape):
    output_image = np.copy(image)
    perimeter = draw.polygon_perimeter(fitted_shape[:, 1], fitted_shape[:, 0])
    draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 255])
    cv2.imwrite('fitted.png', output_image)


shapes = [read_shape(i) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=5)

images = [get_test_image('wing_area', 'cropped', '{}.png'.format(i)) for i in range(4)]

for i in range(4):
    cv2.imwrite('patches_{}.png'.format(i), render_oriented_patches(images[i], shapes[i], (32, 32)))

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

image = get_test_image('wing_area', 'cropped', '0.png')
inference = subspace_shape.infer(image, images, shapes, *shape_model)
for iteration in range(5):
    fitted_shape = next(inference)
    print('Completed {} iteration{}'.format(iteration + 1, 's' if iteration > 0 else ''))
    visualize_fitting(image, fitted_shape)
