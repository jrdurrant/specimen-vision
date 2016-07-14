from vision.segmentation.segmentation import segment_butterfly_contour


def align(images):
    outlines = [segment_butterfly_contour(image) for image in images]
    return outlines
