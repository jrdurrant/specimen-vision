import numpy as np
from skimage.graph import route_through_array
import cv2

def wing_segmentation(mask, wing_left=0.33, wing_right=0.66, crop=0.8):
    mask_crop = np.copy(mask)
    mask_crop[mask_crop == 0] = 128
    height, width = mask_crop.shape

    crop_left, crop_right = int(width * ((1 - crop) / 2)), int(width * ((1 + crop) / 2))

    mask_crop = mask_crop[:, crop_left:crop_right]
    crop_width = crop_right - crop_left + 1

    wing_left = int(width*wing_left - crop_left)
    indices_left, weight = route_through_array(mask_crop, (0, wing_left), (-1, wing_left))
    indices_left = np.array(indices_left).T
    path_crop = np.zeros_like(mask_crop)
    path_crop[indices_left[0], indices_left[1]] = 255

    wing_right = int(width*wing_right - crop_right)
    indices_right, weight = route_through_array(mask_crop, (0, wing_right), (-1, wing_right))
    indices_right = np.array(indices_right).T
    path_crop[indices_right[0], indices_right[1]] = 255

    path = np.zeros((height + 2, width + 2), dtype='uint8')
    path[1:-1, (crop_left + 1):(crop_right + 1)] = path_crop

    path_fill = np.zeros_like(mask)

    cv2.floodFill(path_fill, path, (0, 0), 255)
    cv2.floodFill(path_fill, path, (crop_width - 1, height - 1), 255)

    wing_mask = np.copy(path_fill)
    wing_mask[wing_mask > 0] = 1
    wing_mask = wing_mask * mask

    return wing_mask, (0, crop_left + np.max(indices_left[1])), (np.max(indices_right[1]), width)

if __name__ == '__main__':
    mask = cv2.imread('debug/wing/mask_small.png')[:,:,0]
    mask2, left_wing, right_wing = wing_segmentation(mask)