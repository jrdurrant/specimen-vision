import numpy as np
import cv2
import os
from vision.segmentation.segmentation import segment_wing

debug_folder = 'debug/radial/'

def find_first_row(binary_image, smooth=False, smooth_size=11):
	polar_w = binary_image.shape[1]
	top = np.array([np.min(np.where(binary_image[:,col] > 0)) for col in range(0, polar_w)])
	if smooth:
		half_smooth_size = (smooth_size - 1) / 2
		top[half_smooth_size:-half_smooth_size] = np.convolve(top, np.ones(smooth_size)/smooth_size, mode='valid')
	return top

def centre_of_mass(binary_image):
	h, w = binary_image.shape
	xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))

	x = np.mean((binary_image*xv)[np.where(binary_image > 0)])
	y = np.mean((binary_image*yv)[np.where(binary_image > 0)])

	return y, x

def polar_transformation(image, origin, size, distance_range, angle_range=(0, 2*np.pi)):
	h, w = image.shape[0:2]
	distance = np.linspace(distance_range[0], distance_range[1], size[0])
	angle = np.linspace(angle_range[0],angle_range[1],size[1])
	angle_arr, distance_arr = np.meshgrid(angle, distance)

	xv, yv = cv2.polarToCart(distance_arr, angle_arr)
	xv += origin[1]
	yv += origin[0]

	yv = np.clip(yv, 0, h - 1)
	xv = np.clip(xv, 0, w - 1)

	if len(image.shape) == 3:
		polar_image = image[yv.astype('int32'), xv.astype('int32'), :]
	else:
		polar_image = image[yv.astype('int32'), xv.astype('int32')]
	return polar_image

def remove_noise(binary_image, kernel_size, iterations=1):
	kernel = np.ones((kernel_size, kernel_size),np.uint8)
	return cv2.dilate(cv2.erode(binary_image, kernel), kernel)

def stretch_to_fill(image, edge):
	stretched_image = np.copy(image)
	for col in range(0, polar_w):
		stretched_image[:, col, :][:, np.newaxis, :] = cv2.resize(stretched_image[edge[col]:, col, :][:, np.newaxis, :], (1, image.shape[0]))

	return stretched_image

def find_wing_edge(image):
	h, w = image.shape[0:2]

	medblur = cv2.medianBlur(image.astype('uint8'), 31)
	hsv_image = cv2.cvtColor(medblur, cv2.cv.CV_BGR2HSV)

	mask = cv2.GC_PR_BGD*np.ones((h, w), dtype='uint8')
	for col in range(0, w):
		mask[0, col] = cv2.GC_FGD if hsv_image[0, col, 2] > 64 else cv2.GC_PR_BGD
		mask[h - 1, col] = cv2.GC_BGD

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	cv2.grabCut(hsv_image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

	wing_edge_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	wing_edge_mask[0, :] = 1 # to ensure every column has at least a single 1

	return h - find_first_row(wing_edge_mask[::-1,:])

def wing_limits(binary_image, origin):
	h, w = binary_image.shape
	xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))
	binary_image[np.where(xv > origin[1])] = 0

	wing_dist, wing_angle = cv2.cartToPolar(xv - origin[1], yv - origin[0])

	H = np.histogram((wing_angle[np.where(binary_image > 0)]).ravel(), 10000)
	H_size = np.cumsum(H[0]*1.0/np.sum(H[0]))

	angle_start = H[1][np.where(H_size > 0.02)[0][0]]
	angle_end = H[1][np.where(H_size > 0.92)[0][0]]

	H = np.histogram((wing_dist[np.where(binary_image > 0)]).ravel(), 10000)
	H_size = np.cumsum(H[0]*1.0/np.sum(H[0]))

	dist_start = H[1][np.where(H_size > 0.05)[0][0]]
	dist_end = np.max(H[1])
	return (angle_start, angle_end), (dist_end, dist_start)


if __name__ == '__main__':
	image_name = 'BMNHE_1354027.JPG'

	image = cv2.imread(os.path.join('data', 'segmented_image', 'color', 'male', image_name))
	mask = cv2.imread(os.path.join('data', 'segmented_image', 'mask', 'male', image_name))[:, :, 0]

	mask2, left_wing, right_wing = segment_wing(mask)
	mask2 = mask2 / 255.0

	mask = mask / 255.0

	cv2.imwrite(os.path.join(debug_folder,'color.png'), image)
	cv2.imwrite(os.path.join(debug_folder,'mask.png'), mask*255)

	h, w, dim = image.shape

	polar_h, polar_w = 800, 2000

	y, x = centre_of_mass(mask)
	y = 672.0
	x = 1332.0
	print('Centre of mass at ({:.2f}, {:.2f})'.format(y, x))


	angle_range, distance_range = wing_limits(mask2, (y, x))

	cv2.imwrite(os.path.join(debug_folder,'color_mask.png'), np.tile(mask2[:,:,np.newaxis], (1,1,3))*image)
	cv2.imwrite(os.path.join(debug_folder,'wing_mask.png'), 255*mask2[:,:,np.newaxis])

	image_polar = polar_transformation(image, (y, x), (polar_h, polar_w), distance_range, angle_range)
	mask_polar = polar_transformation(mask2, (y, x), (polar_h, polar_w), distance_range, angle_range)
	cv2.imwrite(os.path.join(debug_folder,'color_polar.png'),image_polar)
	cv2.imwrite(os.path.join(debug_folder,'mask_polar.png'),mask_polar*255)

	mask_polar = remove_noise(mask_polar, 10)

	cv2.imwrite(os.path.join(debug_folder,'mask_polar_no_noise.png'), mask_polar*255)

	top_edge = find_first_row(mask_polar, smooth=True, smooth_size=21)

	stretched_image = stretch_to_fill(image_polar, top_edge)

	cv2.imwrite(os.path.join(debug_folder,'stretched.png'), stretched_image)

	wing_edge = find_wing_edge(stretched_image[:(polar_h/5),:,:])

	stretched_image = stretch_to_fill(image_polar, top_edge + wing_edge)

	cv2.imwrite(os.path.join(debug_folder,'stretched_no_edge.png'), stretched_image)