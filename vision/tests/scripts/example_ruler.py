from vision.ruler_detection.find_scale import ruler_scale_factor
from vision.tests import get_test_image, get_test_path, get_test_folder_images

image = get_test_image('ruler',
                       'measured',
                       'BMNHE_1045431_13499_7ebc4ab5363564654efa82aadc3ac056c4fe88ec.JPG')

pixel_distance = 37.23
real_world_distance = pixel_distance * ruler_scale_factor(image, distance=0.5)
