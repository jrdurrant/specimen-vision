import fnmatch
import os
from nose.tools import nottest
from vision.io_functions import read_image

TEST_DATA = os.path.abspath(os.path.join(__file__, os.pardir, 'test_data'))

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
IMAGE_EXTENSIONS += [ext.upper() for ext in IMAGE_EXTENSIONS]


@nottest
def get_test_path(*path):
    return os.path.join(TEST_DATA, *path)


@nottest
def get_test_image(*path):
    return read_image(get_test_path(*path))


@nottest
def get_test_folder_images(*directory_path):
    files = os.listdir(os.path.join(TEST_DATA, *directory_path))
    images = []
    for extension in IMAGE_EXTENSIONS:
        images += fnmatch.filter(files, '*' + extension)
    image_paths = [os.path.join(*(directory_path + (image,))) for image in images]
    return sorted(image_paths)
