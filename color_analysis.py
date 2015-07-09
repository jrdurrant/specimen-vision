import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2, ClusterError
from collections import namedtuple
import csv
import segmentation
from folder import apply_all_images

Color = namedtuple('Color', ('RGB', 'proportion'))
Segment = namedtuple('Segment', ('name', 'mask', 'num_colors'))

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
    # input_folder = 'data/all_moths/'
    # output_folder = 'data/moths_segmented/'

    # apply_all_images(input_folder=input_folder,
    #                  output_folder=output_folder,
    #                  function=segmentation.segment_image_file)

    input_folder = 'data/moths_segmented/'
    output_folder = 'output/moths_color_analysis/'

    # check all three folders in case there was an error during segmentation
    images_color = [image_file[6:]
                    for image_file
                    in os.listdir(input_folder)
                    if image_file.startswith('color_')]

    images_abdomen = [image_file[8:]
                    for image_file
                    in os.listdir(input_folder)
                    if image_file.startswith('abdomen_')]

    images_wings = [image_file[6:]
                    for image_file
                    in os.listdir(input_folder)
                    if image_file.startswith('wings_')]

    images = [image
              for image
              in images_color
              if image in images_abdomen
              and image in images_wings]

    with open(os.path.join(output_folder,'colors.csv'), 'wb') as csvfile:
        writer = csv.writer(csvfile)

        num_colors = [10, 5]

        headers = ['Specimen']
        for n in num_colors:
            headers += ['Segment', 'Mean Hue', 'Mean Saturation']
            headers += [channel + str(n) for n in range(n) for channel in ['R', 'G', 'B', 'P']]

        writer.writerow(headers)

        for filename in images:
            color = cv2.imread(os.path.join(input_folder, 'color_' + filename))
            abdomen = cv2.imread(os.path.join(input_folder, 'abdomen_' + filename))[:, :, 0]
            wings = cv2.imread(os.path.join(input_folder, 'wings_' + filename))[:, :, 0]

            HSV = cv2.cvtColor(color, cv2.cv.CV_BGR2HSV)

            segments = [Segment('wings', wings, 10),
                        Segment('abdomen', abdomen, 5)]

            current_row = [os.path.splitext(filename)[0]]

            for segment in segments:
                current_row += [segment.name, 
                                np.mean(HSV[:, :, 0][np.where(segment.mask > 128)]),
                                np.mean(HSV[:, :, 1][np.where(segment.mask > 128)])]

                dc = dominant_colors(color.astype('float32'), segment.num_colors, mask=segment.mask)

                output = visualise_colors(dc, 100 * segment.num_colors, 100)
                cv2.imwrite(os.path.join(output_folder,'{}_{}'.format(segment.name, filename)), output)

                for c in dc:
                    current_row += [c.RGB[channel] for channel in range(0, 3)[::-1]] + [c.proportion]

            writer.writerow(current_row)


