import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def remove_ruler(image):
	green = image[:, :, 1]
	plt.hist(np.ravel(green[-1000:, :]))
	plt.show()
	return image

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
	return mask[top:(top + height), left:(left + width)], image[top:(top + height), left:(left + width)]

if __name__ == "__main__":
	image_folder = 'all_images_clean/full/male/'
	output_folder = 'all_images_clean/mask/male/'
	images = (image_name for image_name in os.listdir(image_folder) if os.path.splitext(image_name)[1] == '.JPG')
	for image_name in images:
		test_image = cv2.imread(image_folder + image_name)
		# test_image = cv2.resize(test_image, (206,135))
		segmented_image = segment_butterfly(test_image)
		cv2.imwrite(output_folder+ image_name, segmented_image[0])
		# cv2.imwrite('segmented_small_border/mask_' + image_name, segmented_image[0])