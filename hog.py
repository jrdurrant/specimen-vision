import numpy as np
import cv2

def my_hog(subimage, bin_n=9):
    gx = cv2.Sobel(subimage, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(subimage, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues
    bins = bin_n*ang/(2*np.pi)

    # split values into bins of cell
    low_bins = np.floor(bins)
    high_bins = np.ceil(bins)

    low_bins[np.where(np.less(bins, 1))] = 0
    high_bins[np.where(np.less(bins, 1))] = 0

    low_bins[np.where(np.greater(bins, bin_n - 1))] = bin_n - 1
    high_bins[np.where(np.greater(bins, bin_n - 1))] = bin_n - 1

    mag_low = np.multiply(mag, np.fabs(np.subtract((2*np.pi)*low_bins/bin_n, ang)))
    mag_high = np.multiply(mag, np.fabs(np.subtract((2*np.pi)*high_bins/bin_n, ang)))

    all_bins = np.concatenate((np.ravel(low_bins), np.ravel(high_bins))).astype('int64')
    all_mags = np.concatenate((np.ravel(mag_low), np.ravel(mag_high)))
    hist2 = np.bincount(all_bins, all_mags, bin_n)

    # hist = np.zeros(bin_n)
    # for mag_val, ang_val in zip(np.ravel(mag), np.ravel(ang)):
    #     bin = bin_n*ang_val/(2*np.pi)
    #     if bin < 1:
    #         low_bin = 0
    #         high_bin = 0
    #     elif bin > (bin_n - 1):
    #         low_bin = bin_n - 1
    #         high_bin = bin_n - 1
    #     else:
    #         low_bin = np.floor(bin)
    #         high_bin = np.ceil(bin)

    #     hist[low_bin] += mag_val*abs((2*np.pi)*low_bin/bin_n - ang_val)/(2*np.pi/bin_n)
    #     hist[high_bin] += mag_val*abs((2*np.pi)*high_bin/bin_n - ang_val)/(2*np.pi/bin_n)

    return hist2

test_image = cv2.imread('male_test.jpg')[390:700,422:800,:]
# hist = hog(test_image)
hog_descriptor = cv2.HOGDescriptor()
h = hog_descriptor.compute(test_image)
cv2.imwrite('debug/wing.jpg', test_image)

