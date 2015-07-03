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

def make_cut(costs, start, end):
    cut_y, cut_x = shortest_path(costs, start, end)

    path_crop = np.zeros_like(costs)
    path_crop[cut_y, cut_x] = 255

    path = np.pad(path_crop, (1, 1), mode='constant', constant_values=(0, 0))

    seed_point = tuple(np.argwhere(path[:, 0:1] == 0)[0])
    print seed_point
    mask = np.zeros_like(costs)
    cv2.floodFill(mask, path, seed_point, 1)

    # make sure that the largest side of the cut is filled in
    if np.mean(mask) < 0.5:
        mask = 1 - mask

    return mask

def wing_segmentation(mask, wing_left=0.4, wing_right=0.6):
    height, width = mask.shape

    wing_left = int(width*wing_left)
    wing_right = int(width*wing_right)

    mean_y, mean_x = centre_of_mass(mask)

    mean_x = int(mean_x)

    left_wing_mask = make_cut(costs=mask[:, :mean_x],
                              start=(0, wing_left),
                              end=(height - 1, mean_x - 1))

    right_wing_mask = make_cut(costs=mask[:, mean_x:],
                               start=(0, wing_right - mean_x),
                               end=(height - 1, 0))

    wing_mask = np.hstack((left_wing_mask, right_wing_mask)) * mask
    wing_mask = segmentation.largest_components(wing_mask, 
                                                num_components=2, 
                                                output_bounding_box=False)

    return wing_mask

if __name__ == '__main__':
    with Timer():
        image = cv2.imread('data/full_image/male/BMNHE_1355281.JPG')
        image = cv2.resize(image, (0, 0), fy=0.4, fx=0.4)
        segmented_image, mask = segmentation.segment_butterfly(image, approximate=False)
        cv2.imwrite('debug/wing/color.png', segmented_image)
        cv2.imwrite('debug/wing/segmented.png', mask*255)
        mask2 = wing_segmentation(mask)
        cv2.imwrite('debug/wing/wing_mask.png', mask2)