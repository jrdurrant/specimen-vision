import numpy as np
from skimage.feature import canny
from vision.segmentation.segment import crop_by_saliency, saliency_dragonfly
from vision.tests import get_test_image
from vision.measurements import subspace_shape

src = np.zeros((11, 2))
src[:, 0] = np.arange(-5, 6)
src[:, 1] = np.power(src[:, 0], 2)

dst = np.zeros((11, 2))
dst[:, 0] = np.arange(-5, 6) + 3
dst[:, 1] = np.power(src[:, 0], 2) / 12

mu, phi, sigma2 = subspace_shape.learn((src, dst))

# plt.plot(src[:, 0], src[:, 1], 'r')
# plt.plot(dst[:, 0], dst[:, 1], 'g')

# avg = mu + phi @ (-0.75 * np.ones((1, 1)))
# avg = avg.reshape(-1, 2)
# plt.plot(avg[:, 0], avg[:, 1], 'b')

# plt.show()

image = np.zeros((100, 100), dtype=np.bool)
j = np.arange(20, 80)
i = np.power(j - 50, 2) / 10
image[90 - i.astype(np.int), j] = True

subspace_shape.infer(image, mu, phi, sigma2)
