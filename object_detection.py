import numpy as np
import cv2
from functools import total_ordering

@total_ordering
class Box(object):
	def __init__(self, y, x, score):
		self.y = y
		self.x = x
		self.score = abs(score)

	def __lt__(self, other):
		return self.score < other.score
	def __eq__(self, other):
		return self.score == other.score

def overlap(box1, box2, win_height, win_width, win_area):
	y1, x1 = box1.y, box1.x
	y2, x2 = box2.y, box2.x
	return max(win_height - abs(y1 - y2), 0)*max(win_width - abs(x1 - x2), 0)*1.0/win_area

def nms(boxes, win_height, win_width, min_overlap=0.5):
	num_boxes = len(boxes)
	boxes = sorted(boxes, reverse=True)

	win_area = win_height*win_width

	maximum_boxes = []

	while len(boxes) > 0:
		current_box = boxes[0]
		maximum_boxes.append(current_box)
		boxes = [box for box in boxes if overlap(box, current_box, win_height, win_width, win_area) <= min_overlap]
	return maximum_boxes

def visualize_boxes(boxes, image, win_height, win_width, normalise=True):
    # image_boxes = np.tile(image[:,:,np.newaxis], (1,1,3))
    image_boxes = np.copy(image)

    if len(boxes) > 0:
        boxes.sort()

        score_min, score_max = boxes[0].score, boxes[-1].score
        score_range = score_max - score_min
        normalise = score_range != 0 if normalise else False

        for box in boxes:
            y, x, score = box.y, box.x, box.score
            # print score
            quality = (score - score_min)/score_range if normalise else 1
            cv2.rectangle(image_boxes, (x, y), (x + win_width, y + win_height), (0,255*quality,255*(1-quality)), 1)
    return image_boxes

win_height = 48
win_width = 48
boxes = np.load('boxes.npy')
nms_boxes = nms(boxes, win_height, win_width)