import numpy as np
import cv2
from skimage.graph import MCP_Geometric
import os
from vision import Box, Contour
from operator import attrgetter


def largest_components(binary_image, num_components=1):
    binary_image, contours, hierarchy = cv2.findContours(binary_image.astype('uint8'),
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_NONE)
    contours = [Contour(points) for points in contours]
    return sorted(contours, key=attrgetter('area'), reverse=True)[:num_components]


def saliency_map(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saliency = 0.25 * hsv_image[:, :, 2] + 0.75 * hsv_image[:, :, 1]
    return saliency.astype('float32')


def segment_butterfly(image, saliency_threshold=100, border=10):
    image_height, image_width = image.shape[:2]

    saliency = saliency_map(image)
    blur = cv2.GaussianBlur(saliency, (5, 5), 0).astype(np.uint8)
    _, mask = cv2.threshold(blur, saliency_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    butterfly_contour = largest_components(mask, num_components=1)[0]
    mask = butterfly_contour.draw(image=np.zeros_like(mask), filled=True)

    bounding_box = butterfly_contour.bounding_box
    bounding_box.grow(border)
    image_extents = Box(0, 0, image_width, image_height)
    bounding_box &= image_extents

    return image[bounding_box.indices], mask[bounding_box.indices].astype(np.uint8)


def centre_of_mass(grayscale_image):
    weight = grayscale_image
    h, w = weight.shape
    yv, xv = np.mgrid[:h, :w]

    total_weight = np.sum(weight)

    x = np.sum(weight * xv) / total_weight
    y = np.sum(weight * yv) / total_weight

    return y, x


def shortest_path(costs, start, end):
    offsets = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
    mcp = MCP_Geometric(costs, offsets)
    mcp.find_costs([start], [end])

    indices = np.array(mcp.traceback(end))

    return indices[:, 0], indices[:, 1]


def min_cost_cut(costs, start, end, seed_point, mask_cost=10):
    cut_y, cut_x = shortest_path(costs * (mask_cost - 1) + 1, start, end)

    path_crop = np.zeros_like(costs)
    path_crop[cut_y, cut_x] = 255

    path = np.pad(path_crop, (1, 1), mode='constant', constant_values=(0, 0))

    wing_mask = np.zeros_like(costs)
    # pass a copy of the path image since it is altered by floodFill
    cv2.floodFill(wing_mask, np.copy(path), seed_point, 1)

    return wing_mask, path


def segment_wing(mask, wing_left=0.4, wing_right=0.6, crop=0.3):
    height, width = mask.shape

    mean_y, mean_x = centre_of_mass(mask)

    crop_width = int(width * crop)

    print(mean_y, mean_x)

    wing_left = int(mean_x * (0.5 + wing_left))
    wing_right = int(mean_x * (0.5 + wing_right))

    mean_x = int(mean_x)

    left_wing_mask, left_wing_path = min_cost_cut(costs=mask[:, crop_width:mean_x],
                                                  start=(0, wing_left - crop_width),
                                                  end=(height - 1, mean_x - 1 - crop_width),
                                                  seed_point=(0, height - 1))

    left_wing_path = np.pad(left_wing_path[1:-1, 1:-1],
                            pad_width=((0, 0), (crop_width, width - mean_x)),
                            mode='constant',
                            constant_values=0)

    right_wing_mask, right_wing_path = min_cost_cut(costs=mask[:, mean_x:-crop_width],
                                                    start=(0, wing_right - mean_x),
                                                    end=(height - 1, 0),
                                                    seed_point=(width - mean_x - crop_width - 1, height - 1))

    right_wing_path = np.pad(right_wing_path[1:-1, 1:-1],
                             pad_width=((0, 0), (mean_x, crop_width)),
                             mode='constant',
                             constant_values=0)

    wing_mask = np.hstack((left_wing_mask, right_wing_mask))
    wing_mask = mask * np.pad(wing_mask,
                              pad_width=((0, 0), (crop_width, crop_width)),
                              mode='constant',
                              constant_values=1)

    left_wing_contour, right_wing_contour = largest_components(wing_mask, num_components=2)
    wings_bounding_box = left_wing_contour.bounding_box | right_wing_contour.bounding_box
    wing_mask = np.zeros((wings_bounding_box.height, wings_bounding_box.width))

    wing_mask = left_wing_contour.draw(image=wing_mask, filled=True)
    wing_mask = right_wing_contour.draw(image=wing_mask, filled=True)

    return wing_mask, (left_wing_path, right_wing_path), (mean_y, mean_x)


def segment_image_file(file_in, folder_out):
    image = cv2.imread(file_in)
    segmented_image, segmented_mask = segment_butterfly(image, saliency_threshold=64)

    filename = os.path.basename(file_in)

    file_out = os.path.join(folder_out, 'color_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, segmented_image * (segmented_mask[:, :, np.newaxis] / 255))

    file_out = os.path.join(folder_out, 'full_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, segmented_mask)

    wing_mask = segment_wing(segmented_mask)[0] / 255
    file_out = os.path.join(folder_out, 'wings_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, 255 * wing_mask)

    body_mask = np.greater(segmented_mask, wing_mask).astype('uint8')
    body_contour = largest_components(body_mask * 255, num_components=1)[0]
    body_image = body_contour.draw(filled=True, crop=True)
    file_out = os.path.join(folder_out, 'abdomen_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, 255 * body_image)
