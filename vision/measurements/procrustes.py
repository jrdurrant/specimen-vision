import numpy as np
from skimage.transform import estimate_transform, matrix_transform


def transform_to_template(shape, template):
    transformation = estimate_transform('similarity', shape, template)
    return matrix_transform(shape, transformation.params)


def normalise_shape(shape):
    centre = np.mean(shape, axis=0)
    shape -= centre

    distances_from_centre = np.sqrt(np.sum(np.power(shape, 2), axis=1))
    return shape / np.mean(distances_from_centre)


def mean_shape(shapes):
    all_shapes = np.stack(shapes, axis=2)
    return np.mean(all_shapes, axis=2)


def generalized_procrustes(shapes):
    mean_template = normalise_shape(shapes[0])
    transformed_shapes = shapes

    for iteration in range(10):
        transformed_shapes = [transform_to_template(shape, mean_template) for shape in transformed_shapes]
        mean_template = mean_shape(transformed_shapes)
    return transformed_shapes
