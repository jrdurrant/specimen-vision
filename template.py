import numpy as np
import cv2
from scipy import signal

test_image = cv2.imread('male_test2.jpg').astype('float32')
test_image = cv2.resize(test_image, (0,0), fx=0.2, fy=0.2)
wing_image = cv2.imread('male_test.jpg')[390:700,422:800,:].astype('float32')
wing_image = cv2.resize(wing_image, (0,0), fx=0.2, fy=0.2)

test_gx = cv2.Sobel(test_image[:,:,1], cv2.CV_32F, 1, 0)
test_gy = cv2.Sobel(test_image[:,:,1], cv2.CV_32F, 1, 0)
test_gradient, _ = 0.5*np.add(cv2.cartToPolar(test_gx, test_gy), test_image[:,:,1])

wing_gx = cv2.Sobel(wing_image[:,:,1], cv2.CV_32F, 1, 0)
wing_gy = cv2.Sobel(wing_image[:,:,1], cv2.CV_32F, 1, 0)
wing_gradient, _ = 0.5*np.add(cv2.cartToPolar(wing_gx, wing_gy), wing_image[:,:,1])

corr = signal.correlate2d(test_gradient, wing_gradient, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape)

theight, twidth, _ = wing_image.shape

composite_image = np.copy(test_image)
composite_image[y:(y + theight), x:(x + twidth), :] = wing_image
cv2.imwrite('comp.png', composite_image)
print y, x