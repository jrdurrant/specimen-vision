from skimage.transform import hough_line
from skimage.morphology import skeletonize
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import peakutils
from statsmodels.tsa.stattools import acf
from scipy.ndimage import convolve

def find_edges(image):
	threshold_val, binary_image = cv2.threshold(image[:, :, 1], 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return skeletonize(1 - binary_image / 255)

def hough_transform(binary_image):
	hspace, angles, distances = hough_line(binary_image, theta=np.linspace(0, np.pi, 180, endpoint=False))
	return hspace.astype(np.float32), angles, distances

def find_grid(hspace_angle):
	n = hspace_angle.shape[0]
	# autocorrelation = np.correlate(hspace_angle, hspace_angle, mode='full')[(n - 1):]
	autocorrelation = acf(hspace_angle, nlags=100)
	peaks = peakutils.indexes(autocorrelation)
	start = np.argmax(autocorrelation[5:]) + 5
	separation = np.median(np.diff(peaks))
	return start, separation

image = cv2.imread('BMNHE_500606.JPG')
height, width = image.shape[:2]
edges = find_edges(image)

cv2.imwrite('edges.png', edges * 255.0)

hspace, angles, distances = hough_transform(edges)

start, separation = find_grid(hspace[:, 0])
start = int(start)
separation = int(start)

max_gap = 5
min_length = 5