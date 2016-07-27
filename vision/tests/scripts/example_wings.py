import cv2
from skimage.feature import canny
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image

image = get_test_image('wing_area', 'pinned', 'DSC_0001 (37).jpg')
saliency_map = saliency_dragonfly(image)
cv2.imwrite('saliency.png', saliency_map)

wings_image = image[crop_by_saliency(saliency_map)]
cv2.imwrite('wings.png', wings_image)
cv2.imwrite('wing_edge.png', 255 * canny(wings_image[:, :, 1], 2))
