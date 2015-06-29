import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
import timeit

img = cv2.imread('all_images_clean/segmented/female/BMNHE_1353164.JPG')
seg = cv2.imread('all_images_clean/mask/female/BMNHE_1353164.JPG')

h, w, _ = img.shape

kernel = np.ones((h/100,h/100),np.uint8)
fg = cv2.erode(seg, kernel, iterations=1)[:,:,0]
bg = cv2.dilate(seg, kernel, iterations=1)[:,:,0]

mask = cv2.GC_PR_FGD*np.ones((h, w), dtype='uint8')
mask[np.where(fg > 0)] = cv2.GC_FGD
mask[np.where(bg < 255)] = cv2.GC_BGD

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

cv2.imwrite('boundaries.png', mask/cv2.GC_FGD)

cv2.grabCut(img, mask, None, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_MASK)

# plt.figure()
# plt.imshow(mask)
# plt.colorbar()
# plt.show()

mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img_seg = img*mask[:,:,np.newaxis]
cv2.imwrite('grabcut2.png', mask*255)