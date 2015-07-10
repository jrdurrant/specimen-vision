import numpy as np
import cv2
import os
from skimage.graph import MCP_Geometric
from folder import apply_all_images
import pdb

# hack to account for drawContours function not appearing to fill the contour
def fillContours(image, contours):
    cv2.drawContours(image, contours, contourIdx=-1, color=255, lineType=8, thickness=cv2.cv.CV_FILLED)

    h, w = image.shape
    outline = np.zeros((h + 2, w + 2), dtype='uint8')
    outline[1:-1, 1:-1] = image

    image = 255 * np.ones_like(image)
    cv2.floodFill(image, outline, (0, 0), newVal=0)
    return image

def largest_components(binary_image, num_components=1):
    contours, hierarchy = cv2.findContours(binary_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[:num_components]
    filled_image = fillContours(np.zeros_like(binary_image), contours)

    return filled_image, cv2.boundingRect(np.concatenate(contours))

def grabcut(image, foreground, background, iterCount, mode):
    height, width = image.shape[:2]
    mask = cv2.GC_PR_FGD*np.ones((height, width), dtype='uint8')

    mask[np.where(foreground > 0)] = cv2.GC_FGD
    mask[np.where(background < 255)] = cv2.GC_BGD

    backgroundModel = np.zeros((1, 65), np.float64)
    foregroundModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect=None, bgdModel=backgroundModel, fgdModel=foregroundModel, iterCount=iterCount, mode=mode)

    return np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

def edge_refinement(image, mask, num_components=1):
    h, w, _ = image.shape
    kernel_size = h / 100
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    foreground = cv2.erode(mask, kernel, iterations=1)
    foreground = largest_components(foreground, num_components)[0]

    background = cv2.dilate(mask, kernel, iterations=1)

    mask = grabcut(image, foreground, background, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)

    mask_holes_removed = largest_components(mask, num_components=1)[0]

    segmented_image = image * (mask_holes_removed[:, :, np.newaxis] / 255)
    return segmented_image, mask_holes_removed * 255

def saliency_map(image):
    hsv_image = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)
    saliency = 0.25 * hsv_image[:, :, 2] + 0.75 * hsv_image[:, :, 1]
    return saliency.astype('float32')

def crop_within_limits(image, bounding_rect):
    left, top, crop_width, crop_height = bounding_rect
    image_height, image_width = image.shape[:2]

    crop_left, crop_right = max(left, 0), min(left + crop_width, image_width)
    crop_top, crop_bottom = max(top, 0), min(top + crop_height, image_height)

    return image[crop_top:crop_bottom, crop_left:crop_right]

def segment_butterfly(image, saliency_threshold=100, approximate=True, border=10):
    image_height, image_width = image.shape[:2]
    _, mask = cv2.threshold(saliency_map(image), saliency_threshold, 255, cv2.THRESH_BINARY)

    component, bounding_rect = largest_components(mask, num_components=1)

    left, top, width, height = bounding_rect
    bounding_rect = (left - border, top - border, width + 2*border, height + 2*border)
    
    image = crop_within_limits(image, bounding_rect)
    mask = crop_within_limits(mask, bounding_rect)

    if not approximate:
        image, mask = edge_refinement(image, mask)

    return image, mask

def centre_of_mass(binary_image):
    h, w = binary_image.shape
    xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))

    binary_image[np.where(binary_image > 0)] = 1

    x = np.mean((binary_image * xv)[np.where(binary_image > 0)])
    y = np.mean((binary_image * yv)[np.where(binary_image > 0)])

    return y, x

def shortest_path(costs, start, end):
    offsets = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
    mcp = MCP_Geometric(costs, offsets)
    mcp.find_costs([start], [end])

    indices = np.array(mcp.traceback(end))

    return indices[:, 0], indices[:, 1]

def make_cut(costs, start, end, seed_point, mask_cost=10):
    cut_y, cut_x = shortest_path(costs*(mask_cost - 1) + 1, start, end)

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

    wing_left = int(mean_x * (0.5 + wing_left))
    wing_right = int(mean_x * (0.5 + wing_right))

    mean_x = int(mean_x)

    left_wing_mask, left_wing_path = make_cut(costs=mask[:, crop_width:mean_x],
                                              start=(0, wing_left - crop_width),
                                              end=(height - 1, mean_x - 1 - crop_width),
                                              seed_point=(0, height - 1))

    left_wing_path = np.pad(left_wing_path[1:-1, 1:-1],
                            pad_width=((0, 0), (crop_width, width - mean_x)),
                            mode='constant',
                            constant_values=0)

    right_wing_mask, right_wing_path = make_cut(costs=mask[:, mean_x : -crop_width],
                                                start=(0, wing_right - mean_x),
                                                end=(height - 1, 0),
                                                seed_point=(width - mean_x - crop_width - 1, height - 1))

    right_wing_path = np.pad(right_wing_path[1:-1, 1:-1],
                            pad_width=((0, 0), (mean_x, crop_width)),
                            mode='constant',
                            constant_values=0)

    wing_mask = np.hstack((left_wing_mask,right_wing_mask))
    wing_mask = mask * np.pad(wing_mask, 
                              pad_width=((0, 0), (crop_width, crop_width)),
                              mode='constant',
                              constant_values=1)

    wing_mask = largest_components(wing_mask, num_components=2)[0]

    return wing_mask, (left_wing_path, right_wing_path)

def segment_image_file(file_in, folder_out):
    image = cv2.imread(file_in)
    segmented_image, segmented_mask = segment_butterfly(image, 
                                                        saliency_threshold=64,
                                                        approximate=False)

    filename = os.path.basename(file_in)

    file_out = os.path.join(folder_out, 'color_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, segmented_image)

    # file_out = os.path.join(folder_out, 'full_' + filename)
    # file_out = os.path.splitext(file_out)[0] + '.png'
    # cv2.imwrite(file_out, 255*segmented_mask)

    wing_mask = segment_wing(segmented_mask)[0] / 255
    file_out = os.path.join(folder_out, 'wings_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, 255*wing_mask)

    body = np.greater(segmented_mask, wing_mask).astype('uint8')
    body = largest_components(body*255, num_components=1)[0] / 255
    file_out = os.path.join(folder_out, 'abdomen_' + filename)
    file_out = os.path.splitext(file_out)[0] + '.png'
    cv2.imwrite(file_out, 255*body)