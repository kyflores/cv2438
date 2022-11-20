# Multidet
Detect and track contours/blobs, YoloV5 targets, and Apriltags.

## Install
1. `git submodule update --recursive --init`
2. `pip install -r requirements.txt`

## Common detection format
All detectors should support a `detect(img)` function.

Detectors should return a dictionary with at least the following
for each detection type
* `type`, Which detector this originated from
* `id` Could be a tag family, object type
* `color` RGB triple for the box color.
* `corners` of the bounding area of the detection
  * numpy array, 4 rows, 2 columns, shape = (4, 2)

Multiple detections should be returned in a list.

Several detector-specific or post-processing keys may also
exist, but are not guaranteed for every object.
* `distance` (stereo depth estimator, apriltags)
* `pose` (apriltags)
* `confidence` (yolov5)
