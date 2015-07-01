import numpy as np
import cv2

# hack to account for drawContours function not appearing to fill the contour
def fillContour(image, contour):
	cv2.drawContours(image, contour, contourIdx=-1, color=255, lineType=8, thickness=cv2.cv.CV_FILLED)

	h, w = image.shape
	outline = np.zeros((h + 2, w + 2), dtype='uint8')
	outline[1:-1, 1:-1] = image

	image = 255*np.ones_like(image)
	cv2.floodFill(image, outline, (0, 0), newVal=0)
	return image

def largest_component(binary_image):
	contours, hierarchy = cv2.findContours(binary_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	contour = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
	return fillContour(np.zeros_like(binary_image), contour)

def segment_component(image, mask):
	h, w, _ = image.shape

	kernel = np.ones((h/100,h/100),np.uint8)
	foreground = cv2.erode(seg, kernel, iterations=1)[:,:,0]

	foreground = largest_component(foreground)

	background = cv2.dilate(seg, kernel, iterations=1)[:,:,0]

	mask = cv2.GC_PR_FGD*np.ones((h, w), dtype='uint8')
	mask[np.where(foreground > 0)] = cv2.GC_FGD
	mask[np.where(background < 255)] = cv2.GC_BGD

	backgroundModel = np.zeros((1,65),np.float64)
	foregroundModel = np.zeros((1,65),np.float64)

	boundaries = mask / cv2.GC_FGD

	cv2.grabCut(image, mask, None, backgroundModel, foregroundModel, 10, cv2.GC_INIT_WITH_MASK)

	mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
	segmented_image = image * mask[:,:,np.newaxis]
	return segmented_image, mask * 255, boundaries

if __name__ == '__main__':
	img = cv2.imread('debug/radial/color.png')
	seg = cv2.imread('debug/radial/wing_mask.png')

	segmented_image, segmented_mask = segment_componentimg, seg)[0:2]