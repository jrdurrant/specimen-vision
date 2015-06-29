import cv2
import numpy as np
import os
import glob
import timeit
from collections import namedtuple
from object_detection import Box, visualize_boxes, nms

class Category:
    def __init__(self, name, label, image_paths=[]):
        self.name = name
        self.label = label
        self.image_paths = image_paths
        self.num_samples = len(image_paths)

def gabor_kernels(num_orientations, sigma_vals=(1,3), lamd_vals=(9,11), ksize=15, gamma=0.5, psi=0, ktype=cv2.CV_32F):
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / num_orientations):
        for sigma in sigma_vals:
            for lambd in lamd_vals:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype)
                kernels.append(kernel)
    return kernels

def gabor_features(image, mask=None):
    if mask is not None and np.mean(mask) < 170:
        return None

    feats = np.zeros((len(kernels), 2), dtype='float32')
    for k, kernel in enumerate(kernels):
        filtered = cv2.filter2D(image.astype('float32'), kernel=kernel, **gabor_params)
        if mask is not None:
            filtered = filtered[np.where(mask != 0)]
        feats[k, :] = [filtered.mean(), filtered.var()]
    return np.concatenate(feats)

def compute_features(feature_array, label_vector, sample_paths, label):
    for index, current_path in enumerate(sample_paths):
        image = cv2.cvtColor(cv2.imread(current_path), cv2.cv.CV_BGR2HSV)
        image = cv2.resize(image, (win_height,win_width))
        f = gabor_features(image[:, :, 2])
        feature_array[index, :] = np.transpose(f)
        label_vector[index] = label

def find_boxes(image, mask, win_height, win_width):
    h, w = image.shape
    boxes = []
    for y in range(0, h - win_height, 8):
        # print y
        for x in range(0, w - win_width, 8):
            f = gabor_features(image[y:(y + win_height), x:(x + win_width)], mask[y:(y + win_height), x:(x + win_width)])
            if f is not None:
                dist = svm.predict(f, returnDFVal=True)
                if dist < 0:
                    boxes.append(Box(y, x, dist))
    return boxes

def classify(image, mask, svm, win_height, win_width):
    grey_image = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)[:,:,2]

    kernel = np.ones((4,4),np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    boxes = find_boxes(grey_image, mask, win_height, win_width)

    boxes = nms(boxes, win_height, win_width, 0.2)

    image_boxes = visualize_boxes(boxes, image, win_height, win_width)
    scores = [box.score for box in boxes]
    return len(boxes), image_boxes, scores

def test(category, num_images, output_folder):
    num_correct = 0
    for index, image_path in enumerate(category.image_paths[-num_images:]):
        image_name = os.path.basename(image_path)
        image = cv2.resize(cv2.imread(image_path), (0,0), fy=0.1312, fx=0.1312)
        mask = cv2.resize(cv2.imread(os.path.join('data','segmented_image','mask',category.name,image_name)), image.shape[0:2][::-1])[:,:,0]

        num_boxes, image_boxes, scores = classify(image, mask, svm, win_height, win_width)
        
        predicted_class = 'male' if num_boxes > 0 else 'female'

        print index
        # print scores
        # print('Expected class: {}'.format(category.name))
        # print('Predicted class: {} [{} box{} detected]\n'.format(predicted_class, num_boxes, 'es' if num_boxes != 1 else ''))

        cv2.imwrite(os.path.join(output_folder, category.name, '{}_{}'.format(predicted_class, image_name)), image_boxes)

        if predicted_class == category.name:
            num_correct += 1

    print('{} correct out of {} [{:.2%}]'.format(num_correct, num_images, num_correct*1.0/num_images))

svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=2.67, gamma=5.383)
gabor_params = dict(ddepth=cv2.CV_32F, borderType=cv2.BORDER_WRAP)

win_height = 48
win_width = 48

kernels = gabor_kernels(8)

if __name__ == '__main__':
    folders = ['all_images_clean/cropped_wing/female/', 'all_images_clean/cropped_wing/male/']

    categories = []
    for label, folder_path in enumerate(folders):
        image_paths = glob.glob(os.path.join(folder_path, '*.JPG'))
        category_name = os.path.basename(os.path.normpath(folder_path))
        categories.append(Category(category_name, label, image_paths))

    num_images = sum(category.num_samples for category in categories)
    num_features = len(kernels)*2
    features = np.zeros((num_images, num_features), dtype='float32')
    labels = -1*np.ones((num_images, 1), dtype='int32')

    num_samples_processed = 0
    for category in categories:
        print('Number of {} samples: {}'.format(category.name, category.num_samples))
        compute_features(features[num_samples_processed:, :], labels[num_samples_processed:], category.image_paths, category.label)
        num_samples_processed += category.num_samples

    svm = cv2.SVM()
    svm.train(features, labels, params=svm_params)

    # Run testing
    scores = np.zeros((num_images, 3))
    for index in range(0, features.shape[0]):
        scores[index, :] = [labels[index], svm.predict(features[index, :]), 1 - np.abs(scores[index, 0] - scores[index, 1])]
    print('Accuracy = {:.2%}\n'.format(np.mean(scores[:,2])))

    folders = ['all_images_clean/segmented/female/', 'all_images_clean/segmented/male/']

    categories = []
    for label, folder_path in enumerate(folders):
        image_paths = glob.glob(os.path.join(folder_path, '*.JPG'))
        category_name = os.path.basename(os.path.normpath(folder_path))
        categories.append(Category(category_name, label, image_paths))

    start = timeit.default_timer()
    
    test(categories[0], 346, 'classified')
    test(categories[1], 375, 'classified')

    stop = timeit.default_timer()
    print('Elapsed time is {:.2f}s'.format(stop - start))
    svm.save('svm.xml')