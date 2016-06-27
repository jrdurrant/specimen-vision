import numpy as np
import cv2
import segmentation
import csv
import glob
import os
import timeit

def wing_length(wing_mask, wing_path):
    # assume all wings are 'right' wings, 'left' wings should be mirrored; computed lengths will not be changed by this
    wing_length = 0
    height, width = wing_mask.shape[:2]

    i, j = np.where(wing_path == 255)

    path_indices = [(y, x) for (y, x) in sorted(zip(i, j), key=lambda ind: ind[0]) if y > 0 and y < (height - 1) and x > 0 and x < (width - 1)] 

    cut_y, cut_x = [(y, x) for (y, x) in path_indices if np.sum(wing_mask[(y - 1):(y + 2), x:(x + 2)]) >= (3 * 255)][0]
    wing = wing_mask[:cut_y, cut_x:]
    xv, yv = np.meshgrid(np.arange(cut_x, width), np.arange(0, cut_y))

    xv, yv = xv[np.where(wing == 255)], yv[np.where(wing == 255)]

    if xv.size != 0 and yv.size != 0:
        distance, _ = cv2.cartToPolar(xv.astype('float32') - cut_x, yv.astype('float32') - cut_y)

        index = np.argmax(distance)
        wing_length = distance[index, 0]

    return wing_length

def get_specimen_ids(filename):
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)

        specimen_ids = []

        for row in reader:
            sex = row[3]
            if sex == 'male' or sex == 'female':
                ids = glob.glob(os.path.join('data','full_image',sex,'*{}*'.format(row[0])))
                if ids:
                    if len(ids) > 1:
                        ids = sorted(ids, key=len)
                    specimen_ids.append((row[0], ids[0]))

    return specimen_ids


if __name__ == '__main__':
    specimens = get_specimen_ids('data/H_comma_wing_lengths.csv')

    with open('computed_wing_lengths.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Specimen', 'Right', 'Left'])

    with open('computed_wing_lengths.csv', 'rU') as csvfile:
        reader = csv.reader(csvfile)
        specimen_ids = [row[0] for row in reader]
        specimens = [specimen for specimen in specimens if specimen[0] not in specimen_ids]

    for specimen_id, filename in specimens:
        with open('computed_wing_lengths.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)

            start = timeit.default_timer()
            image = cv2.imread(filename)

            height, width = image.shape[:2]

            segmented_image, segmented_mask = segmentation.segment_butterfly(image, 
                                                                             saliency_threshold=96,
                                                                             approximate=False)

            wing_mask, wing_paths, centre_of_mass = segmentation.segment_wing(segmented_mask)

            left_wing_length = wing_length(wing_mask[:, centre_of_mass[1]::-1], wing_paths[0][:, centre_of_mass[1]::-1])
            right_wing_length = wing_length(wing_mask[:, centre_of_mass[1]:], wing_paths[1][:, centre_of_mass[1]:])
            stop = timeit.default_timer()
            print('{}: Left wing is {:.2f}px, right wing is {:.2f}px [{:.2f}s]'.format(specimen_id, left_wing_length, right_wing_length, stop - start))
            writer.writerow([specimen_id, right_wing_length, left_wing_length])