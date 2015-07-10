import numpy as np
import cv2
import os

def random_crop(image, crop_width=None, crop_height=None):
	h, w, _ = image.shape
	if not crop_width:
		crop_width = np.random.randint(w/4, 3*w/4)
	if not crop_height:
		crop_height = np.random.randint(h/4, 3*h/4)
	left = np.random.randint(0, w - crop_width)
	top = np.random.randint(0, h - crop_height)
	output = image[top:(top + crop_height), left:(left + crop_width), :]
	return output

def crop_folder(path):
	width = 400
	height = 400
	images = (image_name for image_name in os.listdir(path) if os.path.splitext(image_name)[1] == '.JPG')
	for image_name in images:
		test_image = cv2.imread(path + image_name)
		cv2.imwrite(path + image_name, random_crop(test_image, height, width))
		cv2.imwrite(path + 'a' + image_name, random_crop(test_image, height, width))

if __name__ == '__main__':
	crop_folder('all_images_clean/cropped_wing/female/')