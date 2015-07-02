import numpy as np
import cv2
import os
from folder import apply_all_files

def segment_butterfly(image, border=10):
	hsv_image = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)
	mask = 255*np.greater(hsv_image[:,:,1], 100)
	contours, hierarchy = C = cv2.findContours(mask.astype('uint8'), cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_NONE)
	largest_area = 0
	for contour in contours:
		contour_area = cv2.contourArea(contour)
		if contour_area > largest_area:
			bounding_rect = cv2.boundingRect(contour)
			largest_area = contour_area

	left, top, width, height = bounding_rect
	left -= border
	top -= border
	width += border*2
	height += border*2
	return mask[top:(top + height), left:(left + width)],
		   image[top:(top + height), left:(left + width)]

def segment_file(file_in, folder_out):
	image = cv2.imread(file_in)
	segmented_image = segment_butterfly(image)
	cv2.imwrite(os.path.join(folder_out, os.path.basename(file_in)), segmented_image[0])

if __name__ == "__main__":
	apply_all_files(input_folder='data/full_image/male/',
				    output_folder='debug/segment/',
				    function=segment_file)