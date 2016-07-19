import cv2
import numpy as np
from vision import Box


class Contour(object):
    def __init__(self, points):
        self.points = points

    @property
    def bounding_box(self):
        return Box(*cv2.boundingRect(self.points))

    @property
    def area(self):
        return cv2.contourArea(self.points)

    def draw(self, filled=False, image=None, crop=False, color=255):
        if image is None:
            width, height = self.bounding_box.extents
            image = np.zeros((height, width))
            crop = True

        filled_image = cv2.drawContours(image,
                                        [self.points],
                                        contourIdx=-1,
                                        color=color,
                                        lineType=8,
                                        thickness=cv2.FILLED)

        if crop:
            return filled_image[self.bounding_box.indices]
        else:
            return filled_image
