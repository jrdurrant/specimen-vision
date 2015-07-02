import numpy as np
from skimage.graph import MCP_Geometric, route_through_array
import cv2
import segmentation
import timeit

class Timer(object):
    def __enter__(self):
        self.start = timeit.default_timer()
    def __exit__(self, type, value, traceback):
        self.stop = timeit.default_timer()
        print('Elapsed time is {:.2f}s'.format(self.stop - self.start))

def centre_of_mass(binary_image):
    h, w = binary_image.shape
    xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))

    binary_image[np.where(binary_image > 0)] = 1

    x = np.mean((binary_image*xv)[np.where(binary_image > 0)])
    y = np.mean((binary_image*yv)[np.where(binary_image > 0)])

    return y, x

def shortest_path(costs, start, end):
    height, width = costs.shape
    offsets = [[1, -1], [1, 0], [1, 1]]
    mcp = MCP_Geometric(costs, offsets)
    _, traceback = mcp.find_costs([start], [end])
    indices = np.zeros((height, 2), dtype='int32')
    indices[0, :] = end
    for row in range(1, height):
        y, x = indices[row - 1, :]
        indices[row, 0] = indices[row - 1, 0] - offsets[traceback[y, x]][0]
        indices[row, 1] = indices[row - 1, 1] - offsets[traceback[y, x]][1]
    
    return indices[:, 0], indices[:, 1]

def wing_segmentation(mask, wing_left=0.4, wing_right=0.6, crop=0.8, distance_weight=0.5):
    mask_crop = np.copy(mask)
    height, width = mask_crop.shape

    crop_left = int(width * ((1 - crop) / 2))
    crop_right = int(width * ((1 + crop) / 2))

    wing_left = int(width*wing_left - crop_left)
    wing_right = int(width*wing_right - crop_left)

    mean_y, mean_x = centre_of_mass(mask)
    mean_x = int(mean_x - crop_left)

    mask_crop = mask_crop[:, crop_left:crop_right]
    crop_width = crop_right - crop_left + 1

    left_cut_y, left_cut_x = shortest_path(costs=mask_crop[:, :mean_x],
                                           start=(0, wing_left),
                                           end=(height - 1, mean_x - 1))

    path_crop = np.zeros_like(mask_crop)
    path_crop[left_cut_y, left_cut_x] = 255

    right_cut_y, right_cut_x = shortest_path(costs=mask_crop[:, mean_x:],
                                             start=(0, wing_right - mean_x),
                                             end=(height - 1, 0))

    path_crop[right_cut_y, right_cut_x + mean_x] = 255

    path = np.zeros((height + 2, width + 2), dtype='uint8')
    path[1:-1, (crop_left + 1):(crop_right + 1)] = path_crop

    path_fill = np.zeros_like(mask)

    cv2.floodFill(path_fill, path, (0, 0), 255)
    cv2.floodFill(path_fill, path, (crop_width - 1, height - 1), 255)

    wing_mask = np.copy(path_fill)
    wing_mask[wing_mask > 0] = 1
    wing_mask = wing_mask * mask

    wing_mask = segmentation.largest_components(wing_mask, num_components=2, output_bounding_box=False)

    return wing_mask, (0, crop_left + np.max(left_cut_x)), (np.min(right_cut_x), width), path

if __name__ == '__main__':
    with Timer():
        image = cv2.imread('data/full_image/male/BMNHE_1355281.JPG')
        image = cv2.resize(image, (0, 0), fy=0.4, fx=0.4)
        mask, segmented_image = segmentation.segment_butterfly(image, approximate=False)
        cv2.imwrite('debug/wing/color.png', segmented_image)
        cv2.imwrite('debug/wing/segmented.png', mask)
        mask2, left_wing, right_wing, path = wing_segmentation(mask)
        cv2.imwrite('debug/wing/wing_mask.png', mask2)
        cv2.imwrite('debug/wing/path.png', path)