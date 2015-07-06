import cv2
import numpy as np
import os
from scipy.cluster.vq import kmeans2
from collections import namedtuple

Color = namedtuple('Color', ('RGB', 'proportion'))

def dominant_colors(data, num_colors):
    centroids, labels = kmeans2(data, num_colors)

    return centroids, labels

if __name__ == '__main__':
    input_folder = 'data/moths_wings/'
    filename = 'Basiothia_charis_f_MCB002_48.5_r.jpg'

    abdomen = cv2.imread(os.path.join(input_folder, 'abdomen_' + filename))
    color = cv2.imread(os.path.join(input_folder, 'color_' + filename))
    left_wing_front = cv2.imread(os.path.join(input_folder, 'left_wing_front_' + filename))
    left_wing_back = cv2.imread(os.path.join(input_folder, 'left_wing_back_' + filename))
    right_wing_front = cv2.imread(os.path.join(input_folder, 'right_wing_front_' + filename))
    right_wing_back = cv2.imread(os.path.join(input_folder, 'right_wing_back_' + filename))

    i, j = np.where(left_wing_back[:, :, 0] > 250)
    lwb_colors = color[i, j, :]

    i, j = np.where(right_wing_back[:, :, 0] > 250)
    rwb_colors = color[i, j, :]

    colors = np.concatenate((lwb_colors, rwb_colors), axis=0)

    num_colors = 4

    dc, l = dominant_colors(colors.astype('float32'), num_colors)

    order = np.argsort(np.mean(dc, axis=1))[::-1]

    proportions = np.zeros(num_colors)
    for i in range(0, num_colors):
        proportions[i] = np.sum(l == order[i])
    proportions = (num_colors * 100) * proportions / l.shape[0]
    proportions = np.concatenate(([0], np.cumsum(proportions)))

    output = np.zeros((100,num_colors * 100, 3), dtype='float32')
    for i in range(0, num_colors):
        output[:, proportions[i]:proportions[i+1], :] = dc[order[i], :]

    cv2.imwrite('colors.png', output)