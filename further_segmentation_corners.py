import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
import timeit

def shape_context(image, y, x, radius=32, distance_bins=4, angle_bins=12):
	h, w = image.shape

	# start = timeit.default_timer()
	dimension = 2*radius + 1
	xv, yv = np.meshgrid(np.linspace(0, dimension + 1, dimension), np.linspace(0, dimension + 1, dimension))
	distance, angle = cv2.cartToPolar(xv.astype('float32') - radius, yv.astype('float32') - radius)
	image_mask = np.copy(image[(y - radius):(y + radius + 1), (x - radius):(x + radius + 1)])
	mask = np.logical_and(image_mask > 0, distance <= radius)

	sc_quick = np.histogram2d(distance[mask], angle[mask], bins=np.array([distance_bins, angle_bins]), range=np.array([[0, radius], [0, 2*np.pi]]))[0]
	# stop = timeit.default_timer()
	# print stop - start

	# start = timeit.default_timer()
	# sc = np.zeros((distance_bins, angle_bins))
	# for i in range(y - radius, y + radius + 1):
	# 	for j in range(x - radius, x + radius + 1):
	# 		distance, angle = cv2.cartToPolar(np.array(j - x, dtype='float32'), np.array(i - y, dtype='float32'))
	# 		if image[i,j] > 0 and distance <= radius:
	# 			distance_bin = int(distance_bins*distance/radius)
	# 			angle_bin = int(angle_bins*angle/(2*np.pi))
	# 			if distance_bin == distance_bins:
	# 				distance_bin = distance_bins - 1
	# 			if angle_bin == angle_bins:
	# 				angle_bin = angle_bins - 1
	# 			sc[distance_bin, angle_bin] += 1
	# stop = timeit.default_timer()
	# print stop - start

	sc = sc_quick
	edge = np.zeros_like(sc)
	edge[:, 1:] = sc[:, 1:] - sc[:, :-1]
	edge[:, 0] = sc[:, 0] - sc[:, -1]
	left = np.argmin(np.sum(edge, axis=0))
	sc = np.roll(sc, -left, axis=1)
	return sc

def ideal_shape_context(radius=32, target_angle=(np.pi/4), distance_bins=4, angle_bins=12):
	dimension = 2*radius + 1
	image = np.zeros((dimension, dimension), dtype='float32')
	for i in range(0, dimension):
		for j in range(0, dimension):
			distance, angle = cv2.cartToPolar(np.array(radius - j, dtype='float32'), np.array(radius - i, dtype='float32'))
			if distance <= radius and angle >= target_angle:
				image[i,j] = 255

	return shape_context(image, radius, radius, radius, distance_bins, angle_bins)

img = cv2.imread('segmented_small_border/BMNHE_1353971.JPG')
seg = cv2.imread('segmented_small_border/mask_BMNHE_1353971.JPG')

kernel = np.ones((30,30),np.uint8)
fg = cv2.erode(seg, kernel, iterations=1)[:,:,0]
bg = cv2.dilate(seg, kernel, iterations=1)[:,:,0]

h, w, _ = img.shape

mask = cv2.GC_PR_FGD*np.ones((h, w), dtype='uint8')
mask[np.where(fg > 0)] = cv2.GC_FGD
mask[np.where(bg < 255)] = cv2.GC_BGD



bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

cv2.grabCut(img, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

# plt.figure()
# plt.imshow(mask)
# plt.colorbar()
# plt.show()

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_seg = img*mask[:,:,np.newaxis]
cv2.imwrite('mask.png', mask*255)

kernel = np.array([[ 1, 1, 1],
				   [ 1, 0, 1],
				   [ 1, 1, 1]])
connectivity = np.multiply(signal.convolve2d(mask.astype('float32'), kernel, boundary='fill', fillvalue=0, mode='same'), mask.astype('float32'))
connectivity2 = np.copy(connectivity)
connectivity2[np.where(connectivity2 < 1)] = 0
connectivity2[np.where(connectivity2 > (np.sum(kernel) - 1))] = 0
cv2.imwrite('connected.png', connectivity*255/np.sum(kernel))
cv2.imwrite('connected2.png', connectivity2*255)

ideal_sc = ideal_shape_context(target_angle=0.4*np.pi)

good_corner = 1000000000*np.ones_like(connectivity)
white_values = np.where(connectivity2 > 0)
white_indices = zip(white_values[0], white_values[1])
num = 0
total_time = 0
radius = 32
for i, j in white_indices:
	if i >= radius and i < (h - radius) and j >= radius and j < (w - radius):
		num += 1
		# start = timeit.default_timer()
		sc = shape_context(mask, i, j, radius=radius)
		good_corner[i,j] = np.sum(np.power(np.subtract(sc, ideal_sc), 2))
		# stop = timeit.default_timer()
		# total_time += stop - start
		# print 'Pixel %d evaluated in %.4fs. Average time is %.4fs' % (num, stop - start, total_time*1.0/num)

num_corners = 20
corners = np.zeros((num_corners, 3))

corner_output = 128*np.dstack((connectivity, connectivity, connectivity))
corner_sort = np.argsort(good_corner.ravel())
for i in range(0, num_corners):
	y, x = np.unravel_index(corner_sort[i], (h, w))
	corners[i, 0] = y
	corners[i, 1] = x
	corners[i, 2] = good_corner[y, x]
	corner_output[y,x,0] = 0
	corner_output[y,x,1] = 0
	corner_output[y,x,2] = 255


combined_corners = np.arange(0, num_corners)
for i in range(0, num_corners):
	for j in range(i - 1, -1, -1):
		if euclidean(corners[i,:2], corners[j,:2]) < radius/2:
			combined_corners[i] = combined_corners[j]

_, combined_corners = np.unique(combined_corners, return_inverse=True)
num_corners = np.max(combined_corners) + 1
best_corners = np.zeros((num_corners, 3))
for i in range(0, num_corners):
	new_corners = np.where(combined_corners == i)[0]
	best_corners[i,:] = corners[new_corners[np.argmin(corners[new_corners, 2])], :]
	img_seg[best_corners[i, 0], best_corners[i, 1], 0] = 0
	img_seg[best_corners[i, 0], best_corners[i, 1], 1] = 255
	img_seg[best_corners[i, 0], best_corners[i, 1], 2] = 0

gx = cv2.Sobel(np.mean(img, axis=2).astype('uint8'), cv2.CV_32F, 1, 0, ksize=7)
gy = cv2.Sobel(np.mean(img, axis=2).astype('uint8'), cv2.CV_32F, 0, 1, ksize=7)
mag, _ = cv2.cartToPolar(gx, gy)
best_corners = best_corners[:,0:2].astype('int32')
for i in range(0, num_corners):
	if best_corners[i, 1] < w/3.0:
		c = 0
		current = (best_corners[i,0], best_corners[i,1])
		while c < 20 or connectivity[current] > 0:
			img_seg[current[0], current[1], 0] = 0
			img_seg[current[0], current[1], 1] = 255
			img_seg[current[0], current[1], 2] = 0
			next = (current[0], current[1] + 1)
			next_mag = mag[next]
			# print next
			if mag[current[0] - 1, current[1] + 1] < next_mag:
				next = (current[0] - 1, current[1] + 1)
				next_mag = mag[next]
			if mag[current[0] - 1, current[1]] < next_mag:
				next = (current[0] - 1, current[1])
				next_mag = mag[next]
			current = next[:]
			c += 1

cv2.imwrite('corners.png',img_seg)
# corner_kernel = np.ones((radius, radius))
# usan = signal.convolve2d(connectivity/255, corner_kernel, boundary='fill', fillvalue=0, mode='same')
# usan[np.where(usan < (radius**2 * (5.5/8.0)))] = 0
# usan[np.where(usan > (radius**2 * (6.5/8.0)))] = 0
# cv2.imwrite('corners.png', (1-mask)*usan)

# cv2.imwrite('fg.png', np.multiply(fg.astype(np.float32)/255.0, img.astype(np.float32)))
# cv2.imwrite('bg.png', np.multiply(1 - bg.astype(np.float32)/255.0, img.astype(np.float32)))
# cv2.imwrite('mask.png', np.add(fg.astype(np.float32), 0.5*bg.astype(np.float32)))