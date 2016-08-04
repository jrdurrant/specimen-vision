from menpofit.aam import PatchAAM
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
import menpo
from menpo.feature import fast_dsift
import numpy as np
from vision.tests import get_test_image
import csv
import cv2
from skimage import draw


def read_shape(index):
    path = '/home/james/vision/vision/tests/test_data/wing_area/cropped/{}.csv'.format(index)

    vertices = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                vertices.append(row[:2])
    return np.array(vertices, dtype=np.float)


training_images = menpo.io.import_images('/home/james/vision/vision/tests/test_data/wing_area/cropped/',
                                         verbose=True)


patch_aam = PatchAAM(training_images, group='PTS', patch_shape=(35, 35),
                     diagonal=150, holistic_features=fast_dsift,
                     max_shape_components=50, max_appearance_components=150,
                     verbose=True)

fitter = LucasKanadeAAMFitter(patch_aam, n_shape=0.9, n_appearance=0.9)

image = menpo.io.import_image('/home/james/vision/vision/tests/test_data/wing_area/cropped/0.png')
result = fitter.fit_from_shape(image, training_images[1].landmarks['PTS'].lms)

output_image = cv2.imread('/home/james/vision/vision/tests/test_data/wing_area/cropped/0.png')
for i in range(result.n_iters):
    points = result.shapes[i].points
    perimeter = draw.polygon_perimeter(points[:, 0], points[:, 1])
    draw.set_color(output_image, (perimeter[0].astype(np.int), perimeter[1].astype(np.int)), [0, 0, (i / (result.n_iters - 1)) * 255])
cv2.imwrite('fitted.png', output_image)
result.view(render_initial_shape=True, figure_size=(20, 20)).save_figure('fig.png', overwrite=True)
