import numpy as np
import cv2
from scipy.sparse.csgraph import connected_components
from vision.ruler_detection.hough_space import grid_hspace_features
from skimage.feature import canny


def crop_boolean_array(arr):
    B = np.argwhere(arr)
    ystart, xstart = B.min(0)
    ystop, xstop = B.max(0) + 1
    return slice(ystart, ystop), slice(xstart, xstop)


def find_edges(image):
    image_single_channel = image[:, :, 1]
    return canny(image_single_channel, sigma=2)


def best_angles(hspace_entropy):
    angle_scores = np.nanmax(hspace_entropy, axis=-1)
    angle_indices = np.where(np.isnan(np.min(angle_scores, axis=-1)), -1, np.argmin(angle_scores, axis=-1))
    return angle_indices


def find_ruler(image):
    binary_image = find_edges(image) * 255
    cv2.imwrite('binary_image2.png', binary_image)
    hough_spaces, grid_sum = grid_hspace_features(binary_image, grid=16)

    grid_size = hough_spaces.shape[0]

    angle_indices = best_angles(hough_spaces)

    labels = merge_cells(angle_indices)
    n_components = np.max(labels + 1)

    sizes = [np.sum(grid_sum[labels == i])
             for i
             in range(n_components)]
    order = np.argsort(sizes)[::-1]

    height, width = binary_image.shape
    grid_height = int(np.ceil(height / grid_size))
    grid_width = int(np.ceil(width / grid_size))
    mask = np.zeros((height, width), dtype=np.bool)
    for i in range(grid_size):
        for j in range(grid_size):
            grid_i = slice(i * grid_height, (i + 1) * grid_height)
            grid_j = slice(j * grid_width, (j + 1) * grid_width)
            if labels[i, j] == order[0]:
                mask[grid_i, grid_j] = True

    crop = crop_boolean_array(mask)
    return image[crop], mask[crop]


def merge_cells(angle_indices):
    def connection(index_a, index_b):
        if index_a >= 0 and index_b >= 0:
            angle_difference = min((index_a - index_b) % 180, (index_b - index_a) % 180)
            return abs(angle_difference) <= 5
        else:
            return False

    grid_size = len(angle_indices)
    num_grid_elements = grid_size * grid_size

    graph = np.zeros((num_grid_elements, num_grid_elements))
    for i in range(grid_size):
        for j in range(grid_size):
            index = np.ravel_multi_index((i, j), (grid_size, grid_size))

            if i > 0 and connection(angle_indices[i, j], angle_indices[i - 1, j]):
                graph[index, np.ravel_multi_index((i - 1, j), (grid_size, grid_size))] = 1

            if i < (grid_size - 1) and connection(angle_indices[i, j], angle_indices[i + 1, j]):
                graph[index, np.ravel_multi_index((i + 1, j), (grid_size, grid_size))] = 1

            if j > 0 and connection(angle_indices[i, j], angle_indices[i, j - 1]):
                graph[index, np.ravel_multi_index((i, j - 1), (grid_size, grid_size))] = 1

            if j < (grid_size - 1) and connection(angle_indices[i, j], angle_indices[i, j + 1]):
                graph[index, np.ravel_multi_index((i, j + 1), (grid_size, grid_size))] = 1

    n_components, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.reshape(grid_size, grid_size)
