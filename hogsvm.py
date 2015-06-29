import numpy as np
import cv2
import os
import timeit
from read_xml import get_support_vector_from_xml, set_hog_xml_parameters
import tempfile

def HOGDescriptor(win_size=(64,128)):
	hog = cv2.HOGDescriptor()
	with tempfile.NamedTemporaryFile(suffix='.xml') as f:
		hog.save(f.name)
		set_hog_xml_parameters(f.name, win_size=win_size)
		hog.load(f.name)
	return hog

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

win_height = 24
win_width = 24
hog = HOGDescriptor(win_size=(win_height,win_width))

# These values are the only ones supported by OpenCV currently
block_size = (16,16)
cell_size = (8,8)
nbins = 9

num_blocks = (win_height/cell_size[0] - 1, win_width/cell_size[1] - 1)
cells_per_block = (block_size[0]/cell_size[0])*(block_size[1]/cell_size[1])
num_features = num_blocks[0]*num_blocks[1]*cells_per_block*nbins

negative_samples_folder = 'all_images_clean/cropped_wing/female/'
female_images = [negative_samples_folder + image_name for image_name in os.listdir(negative_samples_folder) if os.path.splitext(image_name)[1] == '.JPG']
positive_samples_folder = 'all_images_clean/cropped_wing/male/'
male_images = [positive_samples_folder + image_name for image_name in os.listdir(positive_samples_folder) if os.path.splitext(image_name)[1] == '.JPG']
num_female_images = len(female_images)
num_male_images = len(male_images)
print num_female_images, num_male_images
num_images = num_female_images + num_male_images
features = np.zeros((num_images, num_features), dtype='float32')
labels = -1*np.ones((num_images, 1), dtype='int32')
for i, image_name in enumerate(female_images):
	index = i
	image = cv2.imread(image_name)
	image = cv2.resize(image, (win_height,win_width))
	h = hog.compute(image)
	features[index, :] = np.transpose(h)
	labels[index] = 0

for i, image_name in enumerate(male_images):
	index = i + num_female_images
	image = cv2.imread(image_name)
	image = cv2.resize(image, (win_height,win_width))
	h = hog.compute(image)
	features[index, :] = np.transpose(h)
	labels[index] = 1

svm = cv2.SVM()
svm.train(features, labels, params=svm_params)

with tempfile.NamedTemporaryFile(suffix='.xml') as f:
	svm.save(f.name)
	sv, rho = get_support_vector_from_xml(f.name)

hog.setSVMDetector(sv)

svm.save('svm.xml')
hog.save('hog.xml')
np.save('bias.npy', rho)
