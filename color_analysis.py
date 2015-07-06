import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2, ClusterError
from collections import namedtuple

Color = namedtuple('Color', ('RGB', 'proportion'))
Segment = namedtuple('Segment', ('name', 'mask'))

def dominant_colors(image, num_colors, mask=None):
    image = cv2.cvtColor(image / 255.0, cv2.cv.CV_BGR2Lab)

    if mask is not None:
        i, j = np.where(mask > 250)
        data = image[i, j, :]
    else:
        height, width = image.shape[:2]
        data = np.reshape(image, (height * width, 3))

    # kmeans algorithm has inherent randomness - result will not be exactly the same 
    # every time. Fairly consistent with >= 30 iterations
    centroids, labels = kmeans2(data, num_colors, iter=30)
    counts = np.histogram(labels, bins=range(0, num_colors + 1), normed=True)[0]

    centroids_RGB = cv2.cvtColor(np.reshape(centroids, (centroids.shape[0], 1, 3)), cv2.cv.CV_Lab2BGR)[:, 0, :] * 255.0
    colors = [Color(centroid, count) for centroid, count in zip(centroids_RGB, counts)]
    colors.sort(key=lambda color: np.mean(color.RGB))

    return colors

def visualise_colors(colors, output_width, output_height):
    output = np.zeros((100, output_width, 3), dtype='float32')
    left = 0
    for color in dc:
        right = left + int(color.proportion * output_width)
        output[:, left:right, :] = color.RGB
        left = right

    output[:, right:output_width, :] = colors[-1].RGB

    return output

if __name__ == '__main__':
    input_folder = 'data/moths_wings/'
    output_folder = 'data/moths_colors/'
    filename = 'Basiothia_charis_f_MCB002_48.5_r.png'
    num_colors = 5

    images = [image_file[6:]
              for image_file
              in os.listdir(input_folder)
              if image_file.startswith('color_')]

    for filename in images:
        color = cv2.imread(os.path.join(input_folder, 'color_' + filename))
        abdomen = cv2.imread(os.path.join(input_folder, 'abdomen_' + filename))[:, :, 0]
        forewings = cv2.imread(os.path.join(input_folder, 'forewings_' + filename))[:, :, 0]
        hindwings = cv2.imread(os.path.join(input_folder, 'hindwings_' + filename))[:, :, 0]

        segments = [Segment('abdomen', abdomen),
                    Segment('forewings', forewings),
                    Segment('hindwings', hindwings)]

        for segment in segments:
            dc = dominant_colors(color.astype('float32'), num_colors, mask=segment.mask)

            output = visualise_colors(dc, 100 * num_colors, 100)
            cv2.imwrite(os.path.join(output_folder,'{}_{}'.format(segment.name, filename)), output)