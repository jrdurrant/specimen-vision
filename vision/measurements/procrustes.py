import numpy as np


def normalise_shape(shape):
    centre = np.mean(shape, axis=0)
    distances_from_centre = np.sqrt(np.sum(np.power(shape - centre, 2), axis=1))
    return shape * distances_from_centre / np.mean(distances_from_centre)


def generalized_procrustes(shapes):
    mean_template = normalise_shape(shapes[0])
