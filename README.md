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
