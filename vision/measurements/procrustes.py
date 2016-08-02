import numpy as np
from skimage.transform import estimate_transform, matrix_transform


def transform_to_template(shape, template, transformation_type='similarity'):
    """Returns a transformed shape that is as close as possible to the given
    template under a certain type of transformation.

    Args:
        shape (ndarray): An n x 2 array where each row represents a vertex in
                         the shape. The first column is the x-coordinate and
                         the second column is the y-coordinate.

        template (ndarray): Another shape corresponding to the first input.
                            It must have the same array shape and type, and
                            corresponding rows must respresent corresponding
                            vertices. For example, the vertex respresented by
                            row **i** in the *input* will try to match as
                            closesly as possible to the vertex represented by
                            row **i** in the *template*.

        transformation_type (str): The type of transformation to use when
                                   fitting the shape to the template. The
                                   string must be one of the ones specified by
                                   `skimage.transform.estimate_transform`_.

    Returns:
        ndarray: Transformed shape of the same type and array shape as the
        input shape.

    ..  _skimage.transform.estimate_transform: http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.estimate_transform
    """
    transformation = estimate_transform(transformation_type, shape, template)
    return matrix_transform(shape, transformation.params)


def normalise_shape(shape):
    """Normalises the scale of the shape

    Return a scaled version of the input shape where the average distance of
    a vertex from the centre of the shape is 1.

    Args:
        shape (ndarray): An n x 2 array where each row represents a vertex in
                         the shape. The first column is the x-coordinate and
                         the second column is the y-coordinate.

    Returns:
        ndarray: Transformed shape of the same type and array shape as the
        input shape.
    """
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
