# Computer Vision for biological specimen images
These are a set of tools for extracting metadata from specimen images. They include methods for determining specific properties of specimens, as well as general pre-processing tools. 


## Pre-processing
These can be used as a first step in making a specimen image more usable by automated methods.
#### Segmentation
Separates a specimen from the image background and produces a mask. This gives a silhouette that can be used for analysing the shape, as well as any other analysis that requires the background to be ignored.
#### Determining image scale
Given an image that contains a ruler, this method  determines the scale that the image was taken at. This enables conversion between measurements in pixels and measurements in real world units.


## Metadata
A set of methods for analysing specific properties. Currently this is limited to a set of butterflies and moths.
#### Morphometrics
Determining the area of the specimen's wings, as well as the length of the wing's leading edge.
#### Identifying features
Given examples of a marking, this gives a binary classification on a new image, specifying whether the given marking is present or not.
#### Colour analysis
For an image, or a set of images, this finds the set of dominant colours. This can be helpful for comparison between species as well as simple visualisation. 
