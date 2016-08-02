import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.transform import SimilarityTransform, estimate_transform, matrix_transform
import matplotlib.pyplot as plt


def plot_closest_points(image_points, edge_points, closest_edge_points):
    plt.plot(edge_points[:, 0], edge_points[:, 1], 'r+')
    plt.plot(image_points[:, 0], image_points[:, 1], 'b')
    for im, ed in zip(image_points, closest_edge_points):
        plt.plot([im[0], ed[0]], [im[1], ed[1]], 'g')
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


def find_closest_points(image_points, edge_points):
    pass


def infer(edge_image, mu, phi, sigma2):
    edge_points = np.array(np.where(edge_image)).T
    edge_points[:, [0, 1]] = edge_points[:, [1, 0]]

    translation_estimate = edge_image.shape[1] / 2, edge_image.shape[0] / 2
    scale_estimate = min(edge_image.shape) / 2

    edge_nn = NearestNeighbors(n_neighbors=1).fit(edge_points)
    h = np.zeros((phi.shape[1], 1))
    psi = SimilarityTransform(scale=scale_estimate,
                              translation=translation_estimate)

    for iteration in range(100):
        w = (mu + phi @ h).reshape(-1, 2)
        image_points = matrix_transform(w, psi.params)

        closest_edge_point_indices = edge_nn.kneighbors(image_points)[1].flatten()
        closest_edge_points = edge_points[closest_edge_point_indices]

        if iteration % 20 == 0:
            plot_closest_points(image_points, edge_points, closest_edge_points)

        psi = estimate_transform('similarity', w, closest_edge_points)

        image_points = matrix_transform(w, psi.params)

        closest_edge_point_indices = edge_nn.kneighbors(image_points)[1].flatten()
        closest_edge_points = edge_points[closest_edge_point_indices]

        h = update_h(sigma2, phi, closest_edge_points, mu, psi)

    return image_points
