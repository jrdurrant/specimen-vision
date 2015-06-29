import numpy as np
import cv2
import os
import re
import sys
from functools import total_ordering
import heapq

def classify(image, hog, rho, max_detected=8):
	image_boxes = np.copy(image)
	found = hog.detect(image_boxes, winStride=(1,1))

	if len(found[0]) == 0:
		return 'female', image_boxes, 0

	scores = np.zeros(found[1].shape[0])
	for index, score in enumerate(found[1]):
		scores[index] = found[1][index][0]
	order = np.argsort(scores)

	image_boxes = np.copy(image)
	index = 0
	while index < max_detected and found[1][order[index]] - rho < 0:
		current = found[0][order[index], :]
		x, y = current
		h = hog.compute(image[y:(y + win_height), x:(x + win_width), :])
		colour = (0, 255, 0)
		cv2.rectangle(image_boxes, (x, y), (x + win_width, y + win_height), colour, 1)
		index += 1
	# print 'Number of detected objects = %d' % index

	return 'male' if index > 0 else 'female', image_boxes, index, found[0][order[(index-1):index], :], found[1][order[(index-1):index]]

class Category:
	def __init__(self, name):
		self.name = name
		self.correct_samples = []
		self.incorrect_samples = []

@total_ordering
class Sample:
	def __init__(self, name, score=0, top=0, left=0, bottom=0, right=0):
		self.name = name
		self.crop = [top, left, bottom, right]
		self.score = score
	def __lt__(self, other):
		return self.score < other.score
	def __eq__(self, other):
		return self.score == other.score

hog = cv2.HOGDescriptor()
hog.load('hog.xml')
rho = np.load('bias.npy')[()] # weird syntax - converts 1x1 array to scalar (necessary for I/O)

win_height, win_width = hog.winSize
image_width, image_height = (153,100)

def test():
	images_folder = 'all_images_clean/segmented/'
	categories = [Category(category_name) for category_name in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, category_name))]
	for category in categories:
		input_folder = images_folder + category.name + '/'
		image_names = [image_name for image_name in os.listdir(input_folder) if os.path.splitext(image_name)[1] == '.JPG']
		num_images = len(image_names)
		scores = np.zeros((num_images))
		for index, image_name in enumerate(image_names):
			img = cv2.imread(input_folder + image_name)
			img = cv2.resize(img, (image_width,image_height))
			classification = classify(img, hog, rho, category.name)
			if classification[0] == category.name:
				scores[index] = 1
		print 'Accuracy of {} = {:.2%}'.format(category.name, np.sum(scores)/num_images)


# def retrain():
images_folder = 'all_images_clean/cropped_wing/'
categories = [Category(category_name) for category_name in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, category_name))][::-1]
# categories = [Category('female')]
for category in categories:
	input_folder = images_folder + category.name + '/'
	output_folder = 'classified/%s/' % category.name
	for old_image_name in os.listdir(output_folder):
		if os.path.splitext(old_image_name)[1] == '.JPG':
			os.remove(output_folder + old_image_name)
	print output_folder 
	all_images = [re.match('[^A-Z]*(B[A-Z_]*[0-9]*).*', image_name).group(1) + '.JPG' for image_name in os.listdir(input_folder) if os.path.splitext(image_name)[1] == '.JPG']
	images = []
	[images.append(image) for image in all_images if image not in images]
	input_folder = 'all_images_clean/segmented/' + category.name + '/'
	num_images = len(images)
	print num_images
	scores = np.zeros((num_images))
	for index, image_name in enumerate(images):
		img = cv2.imread(input_folder + image_name)
		img = cv2.resize(img, (image_width,image_height))
		classification = classify(img, hog, rho)
		
		if classification[2] > 0:
			current_samples = []
			for (crop_left, crop_top), current_score in zip(classification[3], classification[4].ravel()):
				current_samples.append(Sample(image_name, current_score, crop_top, crop_left, crop_top + win_height, crop_left + win_width))
		else:
			current_samples = [Sample(image_name)]

		if classification[0] == category.name:
			scores[index] = 1
			category.correct_samples += current_samples
		else:
			for sample in current_samples:			
				if len(category.incorrect_samples) < (num_images): 
					category.incorrect_samples.append(sample)
					heapq._heapify_max(category.incorrect_samples)
				else:
					heapq._heappushpop_max(category.incorrect_samples, sample)
		
		cv2.imwrite(output_folder + classification[0] + '_' + image_name, classification[1])
		print '{} [{} out of {}]'.format(classification[0], index + 1, num_images)

	print 'Accuracy = %.2f%%' % (100.0*np.sum(scores)/num_images)

########################################################################################
# Re train on incorrect results from negative incorrect samples
female_category = next(category for category in categories if category.name == 'female')
images_folder = 'all_images_clean/segmented/'

# TODO - REMOVE A SAMPLE FROM THE SAME IMAGE YOU ARE REPLACING WITH - IF POSSIBLE?
output_folder = 'all_images_clean/cropped_wing/female/'
number_to_delete = len(female_category.incorrect_samples)
for old_image_name in os.listdir(output_folder):
		if number_to_delete > 0 and os.path.splitext(old_image_name)[1] == '.JPG':
			os.remove(output_folder + old_image_name)
			number_to_delete -= 1

incorrect_samples_sorted = []
for i in range(0, len(female_category.incorrect_samples)):
	incorrect_samples_sorted.append(heapq._heappushpop_max(female_category.incorrect_samples, Sample('',-1000)))
for index, incorrect_sample in enumerate(incorrect_samples_sorted):
	image_name_full = incorrect_sample.name
	m = re.match('[^A-Z]*(B[A-Z_]*[0-9]*).*', incorrect_sample.name)
	image_name = m.group(1) + '.JPG'
	image = cv2.resize(cv2.imread(images_folder + 'female/' + image_name), (image_width, image_height))
	cropped_image = image[incorrect_sample.crop[0]:incorrect_sample.crop[2], incorrect_sample.crop[1]:incorrect_sample.crop[3], :]
	cv2.imwrite(('all_images_clean/cropped_wing/female/%d_' % index) + image_name, cropped_image)

# if __name__ == "__main__":
# 	retrain()

