Determining image scale from a ruler
====================================

Given an image such as the one below, there are two things needed to determine the scale factor, enabling conversion between pixel measurements and real world measurements.

..  figure::  images/ruler_test_full.jpg
	:align:	  center
	:width:   80%

	Specimen image

* **A visible ruler in the image**: without this there is no way to automatically anchor the size of the specimen in real world space.
* **Distance between the smallest graduations**: for this example it is 0.5mm.

..  figure::  images/drawing.png
    :align:   center
    :width:   40%

    Ruler graduations

A simple example is given of how this is used on an actual image, with the parameters being as seen above. By manually measuring the distance between the smallest graduations (approximately 7.566 pixels) it is confirmed that this results in a real world distance of 0.5mm, as specified. Note that units remain the same and so it is not necessary to specify them.

..  code:: python

	from skimage.io import imread
	from vision.ruler_detection.find_scale import ruler_scale_factor

	image = imread('specimen_image.jpg')
	scale_factor = ruler_scale_factor(image, distance=0.5)
	pixel_distance = 7.566
	real_distance = pixel_distance * scale_factor

..  code:: python

	>>> print(real_distance)
	0.50002001856
