import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage.transform import SimilarityTransform, estimate_transform, matrix_transform
import matplotlib.pyplot as plt


def learn(points, K=1):
    points = [point_set.flatten() for point_set in points]
    w = np.stack(points, axis=1)
    mu = np.mean(w, axis=1).reshape(-1, 1)
    W = w - mu

    U, L2, _ = np.linalg.svd(np.dot(W, W.T))

    D = 22
    K = 1
    sigma2 = np.sum(L2[(K + 1):(D + 1)]) / (D - K)
    phi = np.dot(U[:, :K], np.sqrt(np.diag(L2[:K]) - sigma2 * np.eye(K)))
    return mu, phi, sigma2


def update_h(sigma2, phi, y, mu, psi):
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


def infer(edge_image, mu, phi, sigma2):
    edge_points = np.array(np.where(edge_image)).T
    edge_points[:, [0, 1]] = edge_points[:, [1, 0]]
    edge_points[:, 1] = 90 - edge_points[:, 1]

    # src = np.zeros((11, 2))
    # src[:, 0] = np.arange(-5, 6)
    # src[:, 1] = np.power(src[:, 0], 2)
    # edge_points = src

    edge_nn = NearestNeighbors(n_neighbors=1).fit(edge_points)
    h = np.zeros((phi.shape[1], 1))
    psi = SimilarityTransform(scale=3, translation=(50, 0))

    for iteration in range(100):
        w = (mu + phi @ h).reshape(-1, 2)
        image_points = matrix_transform(w, psi.params)

        closest_edge_point_indices = edge_nn.kneighbors(image_points)[1].flatten()
        closest_edge_points = edge_points[closest_edge_point_indices]

        if iteration % 5 == 0:
            plt.plot(edge_points[:, 0], edge_points[:, 1], 'r+')
            plt.plot(image_points[:, 0], image_points[:, 1])
            for im, ed in zip(image_points, closest_edge_points):
                plt.plot([im[0], ed[0]], [im[1], ed[1]], 'g')
            plt.show()

        psi = estimate_transform('similarity', w, closest_edge_points)

        image_points = matrix_transform(w, psi.params)

        closest_edge_point_indices = edge_nn.kneighbors(image_points)[1].flatten()
        closest_edge_points = edge_points[closest_edge_point_indices]

        h = update_h(sigma2, phi, closest_edge_points, mu, psi)
        print(h)
    print(image_points)
    print(closest_edge_points)
    color_image = np.tile(255 * image[:, :, np.newaxis], (1, 1, 3))
