import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.transform import SimilarityTransform, estimate_transform, matrix_transform, warp
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import cv2


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


def find_closest_points(current_shape, shapes, images, n=15):
    num_points = current_shape.shape[0]
    num_images = len(images)

    angles = local_angle(current_shape)

    closest_points = -np.ones((num_points, 2))

    template_patches = [[extract_rotated_patch(image, shape[p, [1, 0]], patch_angle - np.pi / 2, (32, 32))
                         for p, patch_angle
                         in enumerate(local_angle(shape))]
                        for image, shape
                        in zip(images, shapes)]

    for p in range(num_points):
        # print('Point {}'.format(p))
        current_angle = angles[p]
        current_location = current_shape[p, :]

        coords = np.stack((np.linspace(-250, 250, n), np.zeros(n)), axis=1)
        tform = SimilarityTransform(rotation=(current_angle))
        aligned_coords = (tform(coords) + current_location).astype(np.int)

        similarity = np.zeros((n, num_images))

        for index, (y, x) in enumerate(aligned_coords):
            new_patch = extract_rotated_patch(images[0], [x, y], current_angle - np.pi / 2, (32, 32))
            for template_index in range(num_images):
                similarity[index, template_index] = compare_ssim(template_patches[template_index][p][:, :, 1], new_patch[:, :, 1])
            if p == 5:
                cv2.imwrite('patch_offset_{}.png'.format(index), np.concatenate((new_patch, template_patches[0][5]), axis=1))
        best_offset_index = np.unravel_index(np.argmax(similarity), (n, num_images))[0]
        # print('Best image is {}'.format(np.unravel_index(np.argmax(similarity), (n, num_images))[1]))
        closest_points[p, :] = aligned_coords[best_offset_index, :]
    return closest_points


def local_angle(shape):
    n = shape.shape[0]
    offset = shape[np.mod(np.arange(0, n) + 1, n), :] - shape[np.mod(np.arange(0, n) - 1, n), :]
    return np.pi - np.arctan2(offset[:, 0], offset[:, 1])


def plot_closest_points(image_points, closest_points):
    plt.plot(image_points[:, 0], image_points[:, 1], 'b')
    for image, close in zip(image_points, closest_points):
        plt.plot([image[0], close[0]], [image[1], close[1]], 'g')
    plt.show()


def learn(points, K=1):
    points = [point_set.flatten() for point_set in points]
    w = np.stack(points, axis=1)
    mu = np.mean(w, axis=1).reshape(-1, 1)
    W = w - mu

    U, L2, _ = np.linalg.svd(np.dot(W, W.T))

    D = mu.shape[0]
    sigma2 = np.sum(L2[(K + 1):(D + 1)]) / (D - K)
    phi = U[:, :K] @ np.sqrt(np.diag(L2[:K]) - sigma2 * np.eye(K))
    return mu, phi, sigma2


def update_h(sigma2, phi, y, mu, psi):
    """Updates the hidden variables using updated parameters.

    This is an implementation of the equation:

..  math::
        \\hat{h} = (\\sigma^2 I + \\sum_{n=1}^N \\Phi_n^T A^T A \\Phi_n)^{-1} \\sum_{n=1}^N \\Phi_n^T A^T (y_n - A \\mu_n - b)
    """
    N = y.shape[0]
    K = phi.shape[1]

    A = psi.params[:2, :2]
    b = psi.translation

    partial_0 = 0
    for phi_n in np.split(phi, N, axis=0):
        partial_0 += phi_n.T @ A.T @ A @ phi_n

    partial_1 = sigma2 * np.eye(K) + partial_0

    partial_2 = np.zeros((K, 1))
    for phi_n, y_n, mu_n in zip(np.split(phi, N, axis=0), y, mu.reshape(-1, 2)):
        partial_2 += phi_n.T @ A.T @ (y_n - A @ mu_n - b).reshape(2, -1)

    return np.linalg.inv(partial_1) @ partial_2


def infer(image, images, shapes, mu, phi, sigma2):
    scale_estimate = min(image.shape[:2]) / 2

    h = np.zeros((phi.shape[1], 1))
    psi = SimilarityTransform(scale=scale_estimate)

    while True:
        w = (mu + phi @ h).reshape(-1, 2)
        image_points = matrix_transform(w, psi.params)

        closest_points = find_closest_points(image_points, shapes, images)

        # if iteration % 2 == 0:
        #     plot_closest_points(image_points, closest_points)

        psi = estimate_transform('similarity', w, closest_points)

        image_points = matrix_transform(w, psi.params)

        closest_points = find_closest_points(image_points, shapes, images)

        h = update_h(sigma2, phi, closest_points, mu, psi)

        w = (mu + phi @ h).reshape(-1, 2)
        image_points = matrix_transform(w, psi.params)

        yield image_points
