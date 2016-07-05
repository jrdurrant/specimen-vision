# Computer Vision for biological specimen images

There are three main types of metadata that can be extracted:

## Object detection
This is designed to pick out a particular marking for a given set of samples.
The main code for this is in gabor.py and object\_detection.py.
Requires a set of positive and negative samples (only binary classification).
The parameters to gabor\_kernels() can be adjusted depending ont he kind of samples being used to see what works best

## Wing length and area
All functions inside shape\_analysis.py

## Colour analysis
Colour analysis for moths can be run using color\_analysis.py. Methods for finding dominant colours and visualising are inside and can be reused for different purposes
For Herbarium sheets, the analysis of chlorophyll levels is done using herbarium\_analysis.py

## Segmentation
Less useful independently, but a utility function that is necessary for shape and colour analysis
