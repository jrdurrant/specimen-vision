import numpy as np
from skimage.feature import canny
from vision.tests import get_test_image
from vision.measurements import subspace_shape, procrustes
import csv
import cv2
from skimage import draw
import menpo
import menpofit


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if len(row) == 2:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


shapes = [read_shape(i) for i in range(4)]
aligned_shapes = procrustes.generalized_procrustes(shapes)

shape_model = subspace_shape.learn(aligned_shapes, K=5)

wings_image = get_test_image('wing_area', 'cropped', 'unlabelled', '9.png')
cv2.imwrite('wings.png', wings_image)
edges = canny(wings_image[:, :, 1], 2.5)
cv2.imwrite('wing_edge.png', 255 * edges)

inference = subspace_shape.infer(edges, *shape_model)
for iteration in range(50):
    fitted_shape = next(inference)

output_image = np.copy(wings_image)
points = fitted_shape[:, [1, 0]]
perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, 255])
cv2.imwrite('wings_template.png', output_image)

training_images = menpo.io.import_images('/home/james/vision/vision/tests/test_data/wing_area/cropped/',
                                         verbose=True)


patch_aam = menpofit.aam.PatchAAM(training_images, group='PTS', patch_shape=(35, 35),
                                  diagonal=150, holistic_features=menpo.feature.fast_dsift,
                                  max_shape_components=50, max_appearance_components=150,
                                  verbose=True)

fitter = menpofit.aam.LucasKanadeAAMFitter(patch_aam, n_shape=0.9, n_appearance=0.9)

image = menpo.image.Image(np.transpose(wings_image[:, :, [2, 1, 0]], (2, 0, 1)))
result = fitter.fit_from_shape(image, menpo.shape.PointCloud(fitted_shape[:, [1, 0]]))

result.view(render_initial_shape=True, figure_size=(20, 20)).save_figure('fig.png', overwrite=True)
