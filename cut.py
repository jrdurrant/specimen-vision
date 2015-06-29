import numpy as np
import cv2
import matplotlib.pyplot as plt

test_image = cv2.imread('binary.png')
edges = cv2.Canny(test_image, 128, 128)
cv2.imwrite('debug/edges.png', edges)

intensity = cv2.cvtColor(test_image, cv2.cv.CV_BGR2HSV)
intensity = np.mean(intensity[:,:,1:], axis=2)

img = np.copy(intensity)
h, w = img.shape

order = np.argsort(np.ravel(img))

color_img = np.copy(test_image)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

n = 5000

markers = cv2.GC_PR_BGD*np.ones((h,w), dtype=np.uint8)

for i in range(0, n):
	y, x = np.unravel_index(order[i], (h, w))
	color_img[y,x,0] = 0
	color_img[y,x,1] = 255
	color_img[y,x,2] = 0
	markers[y,x] = cv2.GC_BGD

for i in range(0, n):
	y, x = np.unravel_index(order[-i], (h, w))
	color_img[y,x,0] = 0
	color_img[y,x,1] = 0
	color_img[y,x,2] = 255
	markers[y,x] = cv2.GC_FGD

cv2.grabCut(test_image, markers, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

plt.figure()
plt.imshow(markers)
plt.colorbar()
plt.show()

# for x, y, radius in lines[0]:
#     cv2.circle(img,(int(x),int(y)),int(radius),(0,255,0))

mask = np.where((markers==2)|(markers==0),0,1).astype('uint8')

cv2.imwrite('debug/extrema.jpg',color_img)
cv2.imwrite('debug/markers.jpg',markers*255)