import numpy as np
import cv2
import os

debug_folder = 'radial_debug'

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
		mask[0, col] = cv2.GC_FGD if hsv_image[0, col, 2] > 128 else cv2.GC_PR_BGD
		mask[h - 1, col] = cv2.GC_BGD

	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)

	cv2.grabCut(hsv_image, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

	wing_edge_mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	wing_edge_mask[0, :] = 1 # to ensure every column has at least a single 1

	return h - find_first_row(wing_edge_mask[::-1,:])

def find_wing_span(binary_image):
	line = sorted([(624, 1162), (1048, 1160)])
	h, w = binary_image.shape
	xv, yv = np.meshgrid(np.arange(0, w), np.arange(0, h))
	position = np.sign(np.subtract((line[1][1] - line[0][1])*(yv - line[0][0]), (line[1][0] - line[0][0])*(xv - line[0][1])))
	wing = np.copy(binary_image)
	wing[np.where(position < 0)] = 0

	a = line[1][0] - line[0][0]
	b = line[1][1] - line[0][1]
	c = line[1][1]*line[0][0] - line[1][0]*line[0][1]

	closest_coords_x = np.true_divide((b*np.subtract(b*xv, a*yv) - a*c), (a**2 + b**2))
	closest_coords_y = np.true_divide((a*np.subtract(a*yv, b*xv) - b*c), (a**2 + b**2))

	clip = (closest_coords_y - line[0][0])*(line[1][0] - line[0][0]) + (closest_coords_x - line[0][1])*(line[1][1] - line[0][1])
	clip = clip/((line[1][0] - line[0][0])**2 + (line[1][1] - line[0][1])**2)

	closest_coords_y[np.where(clip > 1)] = line[1][0]
	closest_coords_y[np.where(clip < 0)] = line[0][0]
	closest_coords_x[np.where(clip > 1)] = line[1][1]
	closest_coords_x[np.where(clip < 0)] = line[0][1]

	wing_dist, wing_angle = cv2.cartToPolar(np.subtract(xv, closest_coords_x), np.subtract(yv, closest_coords_y))

	wing_dist = wing_dist*wing

	H = np.histogram((wing_angle[np.where(wing > 0)]).ravel(), 10000)
	H_size = np.cumsum(H[0]*1.0/np.sum(H[0]))

	angle_start = H[1][np.where(H_size > 0.04)[0][0]]
	angle_end = H[1][np.where(H_size > 0.96)[0][0]]
	print angle_start, angle_end

	wing[np.where((wing_angle < angle_start) | (wing_angle > angle_end))] *= 0.5

	return wing, closest_coords_y, closest_coords_x, wing_dist, wing_angle


if __name__ == '__main__':
	image_name = 'BMNHE_1354014.JPG'

	image = cv2.imread(os.path.join('all_images_clean','segmented','male',image_name))
	mask = cv2.imread(os.path.join('all_images_clean','mask','male',image_name))[:,:,0]/255.0

	cv2.imwrite(os.path.join(debug_folder,'color.png'), image)
	cv2.imwrite(os.path.join(debug_folder,'mask.png'), mask*255)

	h, w, dim = image.shape

	wing, close_y, close_x, wing_dist, wing_angle = find_wing_span(mask)
	cv2.imwrite(os.path.join(debug_folder,'wing_dist.png'),wing_dist*255.0/np.max(wing_dist))
	cv2.imwrite(os.path.join(debug_folder,'wing.png'),wing*255)

	# polar_h, polar_w = 800, 2000

	# y, x = centre_of_mass(mask)
	# print('Centre of mass at ({:.2f}, {:.2f})'.format(y, x))

	# cv2.imwrite(os.path.join(debug_folder,'color_mask.png'), np.tile(mask[:,:,np.newaxis], (1,1,3))*image)

	# angle_range = (2.6, 3.75)
	# distance_range = (1480, 250)

	# image_polar = polar_transformation(image, (y, x), (polar_h, polar_w), distance_range, angle_range)
	# mask_polar = polar_transformation(mask, (y, x), (polar_h, polar_w), distance_range, angle_range)
	# cv2.imwrite(os.path.join(debug_folder,'color_polar.png'),image_polar)
	# cv2.imwrite(os.path.join(debug_folder,'mask_polar.png'),mask_polar*255)

	# mask_polar = remove_noise(mask_polar, 10)

	# cv2.imwrite(os.path.join(debug_folder,'mask_polar_no_noise.png'), mask_polar)

	# top_edge = find_first_row(mask_polar, smooth=True, smooth_size=21)

	# stretched_image = stretch_to_fill(image_polar, top_edge)

	# cv2.imwrite(os.path.join(debug_folder,'stretched.png'), stretched_image)

	# wing_edge = find_wing_edge(stretched_image[:(polar_h/5),:,:])

	# stretched_image = stretch_to_fill(image_polar, top_edge + wing_edge)

	# cv2.imwrite(os.path.join(debug_folder,'stretched_no_edge.png'), stretched_image)