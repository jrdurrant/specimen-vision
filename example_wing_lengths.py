import segmentation
import io_functions
import timeit
import csv
import glob
import os

specimens = io_functions.get_specimen_ids('data/H_comma_wing_lengths.csv')

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