import numpy as np
import logging

logging.basicConfig(filename='ruler.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(levelname)s %(message)s')


def where(array, boolean_array, func, default_value):
    output_array = np.ones_like(array) * default_value
    output_array[boolean_array] = func(array[boolean_array])
    return output_array


def cartesian_to_polar(x, y):
    radius = np.sqrt(np.power(x, 2) + np.power(y, 2))
    theta = np.arctan2(y, x) + np.pi
    return radius, theta


def normalise_by_mean(arr):
    return arr / np.mean(arr)


def get_radial_bins(radius_inner, radius_outer, num_bins):
    bins = np.zeros(num_bins + 1)
    bins[1:] = np.logspace(np.log10(radius_inner), np.log10(radius_outer), num_bins)
    return bins


def shape_context(vertex, vertices, num_bins_log_radius=5, num_bins_theta=12, radius_min=1/8, radius_max=2):
    x = vertices[:, 1] - vertex[1]
    y = vertices[:, 0] - vertex[0]
    distances, angles = cartesian_to_polar(x, y)
    log_distances = where(distances, distances > 0, np.log10, 0)
    bins_radius = get_radial_bins(radius_min, radius_max, num_bins_log_radius)
    bins_theta = np.linspace(0, 2 * np.pi, num_bins_theta + 1)

    h, xedges, yedges = np.histogram2d(log_distances, angles, (bins_radius, bins_theta))
    logging.info(h)
    return h


def shape_context_distance(histogram_i, histogram_j):
    return 0.5 * np.sum(np.power(histogram_i - histogram_j, 2) / (histogram_i + histogram_j))
