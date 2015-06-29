import numpy as np
from skimage.graph import route_through_array
import cv2

# image_original = cv2.imread('data/segmented_image/mask/male/BMNHE_1353167.JPG')[:,:,0]
image_original = cv2.imread('debug/wing/mask_small.png')[:,:,0]
image = np.copy(image_original)
image[image == 0] = 128
h0, w0 = image.shape[:2]

image = image[:,(w0/10):(-w0/10)]
h, w = image.shape[:2]

indices, weight = route_through_array(image, (0, -w0/10 + (1*w0)/3), (-1, (1*w)/3))
indices = np.array(indices).T
path = np.zeros_like(image)
path[indices[0], indices[1]] = 255

indices, weight = route_through_array(image, (0, -w0/10 + (2*w0)/3), (-1, (2*w)/3))
indices = np.array(indices).T
path[indices[0], indices[1]] = 255

path2 = np.zeros((h0+2,w0+2), dtype='uint8')
path2[1:-1,(w0/10 + 1):(w0/10 + 1 + w)] = path

image2 = np.zeros_like(image_original)

cv2.floodFill(image2, path2, (0, 0), 255)
cv2.floodFill(image2, path2, (w - 1, h - 1), 255)

image3 = np.copy(image2)
image3[image3 > 0] = 1
image3 = image3 * image_original

# output = np.dstack((image3[:, :, np.newaxis], path[:, :, np.newaxis], np.zeros((h, w, 1)))).astype('uint8')

cv2.imshow('out',image3)