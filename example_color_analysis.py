import csv
import numpy as np
import os
import segmentation
import io_functions

input_folder = 'data/moths_segmented/'
output_folder = 'output/moths_color_analysis/'

images = io_functions.specimen_ids_from_images(os.listdir(input_folder))

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
        if color is not None and abdomen is not None and wings is not None:
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