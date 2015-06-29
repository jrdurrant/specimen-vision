from scipy.fftpack import dct
import numpy as np
import cv2

def phash(image):
	image = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY).astype('float64')
	full_dct = dct(cv2.resize(image, (32,32)))
	reduced_dct = full_dct[:8,:8]
	avg_value = (np.sum(reduced_dct) - reduced_dct[0,0])/(32*32 - 1)
	return 1*np.greater(reduced_dct, avg_value)