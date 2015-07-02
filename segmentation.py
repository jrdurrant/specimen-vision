import numpy as np
import cv2
import os
from folder import apply_all_images

# hack to account for drawContours function not appearing to fill the contour
def fillContours(image, contours):
	cv2.drawContours(image, contours, contourIdx=-1, color=255,
					 lineType=8, thickness=cv2.cv.CV_FILLED)

	h, w = image.shape
	outline = np.zeros((h + 2, w + 2), dtype='uint8')
	outline[1:-1, 1:-1] = image

	image = 255*np.ones_like(image)
	cv2.floodFill(image, outline, (0, 0), newVal=0)
	return image

def largest_components(binary_image, num_components=1, output_bounding_box=False):
	contours, hierarchy = cv2.findContours(binary_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[:num_components]
	filled_image = fillContours(np.zeros_like(binary_image), contours)

	if output_bounding_box:
		return filled_image, cv2.boundingRect(np.concatenate(contours))
	else:
		return filled_image

def grabcut_components(image, mask, num_components=1):
	h, w, _ = image.shape

	kernel = np.ones((h/100,h/100),np.uint8)
	foreground = cv2.erode(mask, kernel, iterations=1)

	foreground = largest_components(foreground, num_components)

	background = cv2.dilate(mask, kernel, iterations=1)

	mask = cv2.GC_PR_FGD*np.ones((h, w), dtype='uint8')
	mask[np.where(foreground > 0)] = cv2.GC_FGD
	mask[np.where(background < 255)] = cv2.GC_BGD

	backgroundModel = np.zeros((1,65),np.float64)
	foregroundModel = np.zeros((1,65),np.float64)

	cv2.grabCut(image, mask, rect=None, bgdModel=backgroundModel,
				fgdModel=foregroundModel, iterCount=10,
				mode=cv2.GC_INIT_WITH_MASK)

	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	segmented_image = image * mask[:,:,np.newaxis]

	mask_holes_removed = largest_components(mask*255, num_components=1, output_bounding_box=False)
	return segmented_image, mask_holes_removed

def segment_butterfly(image, approximate=True, border=10):
	hsv_image = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)
	mask = 255*np.greater(hsv_image[:,:,1], 100).astype('uint8')

	butterfly_component, bounding_rect = largest_components(mask, num_components=1, output_bounding_box=True)
	left, top, width, height = bounding_rect

	crop_left, crop_right = left - border, left + width + border
	crop_top, crop_bottom = top - border, top + height + border

	mask = mask[crop_top:crop_bottom, crop_left:crop_right]
	image = image[crop_top:crop_bottom, crop_left:crop_right]

	if not approximate:
		mask = grabcut_components(image, mask)[1]

	return mask, image

def segment_image_file(file_in, folder_out):
	image = cv2.imread(file_in)
	segmented_image = segment_butterfly(image, approximate=False)

	file_out = os.path.join(folder_out, os.path.basename(file_in))
	cv2.imwrite(file_out, segmented_image[0])

if __name__ == "__main__":
	apply_all_images(input_folder='data/full_image/male/',
				     output_folder='debug/segment/',
				     function=segment_image_file)