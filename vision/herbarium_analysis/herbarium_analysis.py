import cv2
import numpy as np


def heatmap(grayscale_img):
    color_img = np.tile(grayscale_img[:, :, np.newaxis], (1, 1, 3))

    heat = np.linspace(64, 255, 5)
    colors = np.array([[255, 0, 0],
                       [255, 255, 0],
                       [0, 255, 0],
                       [0, 255, 255],
                       [0, 0, 255]], dtype='float32')

    for channel in range(0, 3):
        color_img[:, :, channel] = np.interp(color_img[:, :, channel], heat, colors[:, channel])

    return color_img


def chlorophyll_concentration(rgb_image):
    return 255 - 0.5 * (rgb_image[:, :, 1] + rgb_image[:, :, 2])


def scale_images_to_range(images, min_value, max_value, min_percentile=0, max_percentile=100):
    all_images = np.concatenate(images)
    minimum = np.percentile(all_images, min_percentile)
    maximum = np.percentile(all_images, max_percentile)
    value_range = maximum - minimum

    for index, image in enumerate(images):
        images[index] = ((image - minimum) / value_range) * max_value + min_value

if __name__ == '__main__':
    filenames = ['debug/BM000886029.tif', 'debug/BM000886590.tif']

    imgs = [cv2.imread(filename)[100:7200, 100:4700, :].astype('float64') for filename in filenames]
    chlorophyll_imgs = [chlorophyll_concentration(img) for img in imgs]
    log_scale_imgs = [np.exp(img/96.0) for img in chlorophyll_imgs]

    scale_images_to_range(log_scale_imgs, 0, 255, min_percentile=0, max_percentile=99)

    for img, filename in zip(log_scale_imgs, filenames):
        cv2.imwrite(filename[:-4] + '_heat.png', heatmap(img))
