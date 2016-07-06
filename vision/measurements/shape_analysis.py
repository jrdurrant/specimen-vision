import numpy as np
import cv2


def wing_length(wing_mask, wing_path):
    # assume all wings are 'right' wings, 'left' wings should be mirrored;
    # computed lengths will not be changed by this
    wing_length = 0
    height, width = wing_mask.shape[:2]

    i, j = np.where(wing_path == 255)

    path_indices = [(y, x)
                    for (y, x)
                    in sorted(zip(i, j), key=lambda ind: ind[0])
                    if y > 0 and y < (height - 1) and x > 0 and x < (width - 1)]

    cut_y, cut_x = [(y, x)
                    for (y, x)
                    in path_indices
                    if np.sum(wing_mask[(y - 1):(y + 2), x:(x + 2)]) >= (3 * 255)][0]

    wing = wing_mask[:cut_y, cut_x:]
    xv, yv = np.meshgrid(np.arange(cut_x, width), np.arange(0, cut_y))

    xv, yv = xv[np.where(wing == 255)], yv[np.where(wing == 255)]

    if xv.size != 0 and yv.size != 0:
        distance, _ = cv2.cartToPolar(xv.astype('float32') - cut_x, yv.astype('float32') - cut_y)

        index = np.argmax(distance)
        wing_length = distance[index, 0]

    return wing_length
