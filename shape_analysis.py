import numpy as np
import cv2
import segmentation

def wing_lengths(wing_masks, wing_paths):
    for path, name in zip(wing_paths, ['left', 'right']):
        i, j = np.where(path == 255)

        path_indices = [(y, x) for (y, x) in sorted(zip(i, j), key=lambda ind: ind[0]) if y > 0 and y < (height - 1) and x > 0 and x < (width - 1)] 

        cut_y, cut_x = [(y, x) for (y, x) in path_indices if np.sum(wing_masks[(y - 1):(y + 2), (x - 1):(x + 2)]) > 0][0]
        
        if name == 'left':
            wing = wing_masks[:cut_y, :cut_x]
            xv, yv = np.meshgrid(np.arange(0, cut_x), np.arange(0, cut_y))
        elif name == 'right':
            wing = wing_masks[:cut_y, cut_x:]
            xv, yv = np.meshgrid(np.arange(cut_x, width), np.arange(0, cut_y))

        xv, yv = xv[np.where(wing == 255)], yv[np.where(wing == 255)]

        distance, _ = cv2.cartToPolar(xv.astype('float32') - cut_x, yv.astype('float32') - cut_y)

        index = np.argmax(distance)
        if name == 'left':
            left_wing_length = distance[index, 0]
        elif name == 'right':
            right_wing_length = distance[index, 0]

    return left_wing_length, right_wing_length


if __name__ == '__main__':
    image = cv2.imread('data/full_image/male/BMNHE_500612.JPG')

    height, width = image.shape[:2]

    # segmented_image, segmented_mask = segmentation.segment_butterfly(image, 
    #                                                                  saliency_threshold=96,
    #                                                                  approximate=False)

    # cv2.imwrite('color.png', segmented_image)
    # cv2.imwrite('mask.png', segmented_mask * 255)

    # np.save('color', segmented_image)
    # np.save('mask', segmented_mask)

    segmented_image = np.load('color.npy')
    segmented_mask = np.load('mask.npy')

    wing_mask, wing_paths = segmentation.segment_wing(segmented_mask)

    left_wing_length, right_wing_length = wing_lengths(wing_mask, wing_paths)
    print('Left wing is {:.2f}px\nRight wing is {:.2f}px'.format(left_wing_length, right_wing_length))

    cv2.imwrite('wings.png', wing_mask)