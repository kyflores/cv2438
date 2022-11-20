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
* `sort_xyxy`, xmin, xmax, ymin, ymax format for a bounding box.
  * This format is required by the SORT tracker, and is mostly
    redunant with `corners`

Multiple detections should be returned in a list.

Several detector-specific or post-processing keys may also
exist, but are not guaranteed for every object.
* `distance` (stereo depth estimator, apriltags)
* `pose` (apriltags)
* `confidence` (yolov5)
* `sort_id` (SORT)

## Camera configuration
Most USB webcams have a few configurable parameters. To minimize
motion blur we want to operate at the fastest exposure time possible,
even if that doesn't always translate to more frames delivered per second.

In this context is a device number corresponding to /dev/videoX.
Here's an example command set for configuring the Logitech C310.
Pass `-C` instead of `-c` to read a property instead of set it.
```
v4l2-ctl --list-devices # Find your /dev/videoX number
v4l2-ctl --all -d device # Show all configuration properties
v4l2-ctl -d 3 -c auto_exposure=1
v4l2-ctl -d 3 -c gain=200
v4l2-ctl -d 3 -c exposure_time_absolute=250
```

OpenCV can configure these parameters with the `cv2.CAP_PROP_*` fields,
but just from experimenting with it, it doesn't seem reliable. Setting
the gain for instance changes the number reported by `v4l2-ctl` but
doesn't affect the image.

## TODOs
* Improve mapping SORT boxes back to detect boxes. Current method allows
double assignment, and this seems bugged
* Add blob detector module. Kind of questionable b/c of how much scene
specific tuning and filtering it needs.
* Integrate `robotpy` for network tables support
* Setup the camera framerate automatically, maybe by v4l2-ctl call
* Add Yolo OpenVINO backend support for inference on Intel targets
* Support streaming the augmented camera feed to the driver station
* Add stereo depth map if second camera is available
