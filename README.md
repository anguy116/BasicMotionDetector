# BasicMotionDetector

This assumes that the program is fed two frames of a given video.
Will highlight the object in motion in white and leave the background/unmoving objects black.

## Tech-stack
- Unsupervised ML (K-means)
- Thresholding Gray-Scale values
- Using fastNlDenoising to reduce noise

## Setup
1. Slice two frames of an image and convert it to bitmap
2. Run `python setup.py build_ext --inplace` to compile Cython script locally
3. `python`
   1. `from motionDetector import main; main()`


## Theory

### Process
The objective of this was to figure out how use two frames from a video to do some basic
motion detection.

The idea was that given that we convert RGB -> Gray-scale we can measure the difference in values
and if the value is above a given threshold we can determine that as movement on a binary level.

0 (black) for still | 255 (white) for movement.

There are some problems that arise with this naive/archaic method of thresholding as small
changes in RGB when converted to Gray-scale may result in a larger differnce than expected.

I decided to use a combination of what I previously learned on quantization as well as a 
method from the openCV library to help reduce such noise.

### Thought process
1. We could leverage the fact that quantization is a simple and effective way of reducing the large amount of unique colours while still being descriptive to the original image (given a good value of K)
   1. Major objects such as cars/people would still be "present" and movement between the two images could capture this.
2. At the end of the day there still could be noise. Using something like FastNLMeansDenoising can reduce the "importance" of tiny clusters of colour resulting from differencing two frames
3. Thresholding shouldn't be modified as the data engineering should account for the variable factors, handled in steps 1 and 2

## FastNLMeansDenoising

As opposed to other denoising algorithms such as gaussian mean, and weighted average.
NL stands for Non-Local, this refers to the fact that algorithm will take the average of several
windows that look similar to a given pixel's neighborhood and use that to denoise the image.

If given enough frames we could calculate the distribution of every pixel in order to find the `noise` and subtract it from the true pixel color.
Since we are only give too little frames/just one this is quite impossible.

Obviously, this takes more time compared to other methods but results in a more effective noise reduction.

#References
http://amroamroamro.github.io/mexopencv/opencv/non_local_means_demo.html

https://en.wikipedia.org/wiki/K-means_clustering

https://en.wikipedia.org/wiki/Thresholding_(image_processing)