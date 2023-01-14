# lumine
Prototype multi object detector and tracker suite for FRC

## Install
Preferably in a virtualenv...
```
pip install -r requirements.txt
```
And get `v4l2-ctl` from your package manager.

This project is only tested on linux.
`requirements.txt` is a mess right now, future work will try to pair down
the dependencies needed.

## Embedded Hardware Support (goals)
* CPU, with pytorch and OpenVINO backends.
* Intel IGPs with OpenVINO
* (eventually) NVIDIA Jetson with TensorRT model conversion

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

## Prepping OpenVINO models
This project uses OpenVINO for inference on Intel integrated graphics.
The easiest way to get the Ultralytics/YoloV5 model in OpenVINO format is to
simply request it when calling export.
`python export.py --weights yolov5s.pt --include openvino`
Models exported this way appear to be the standard float32 version.

The hardware may also take advantage of float16 for a performance improvement
with minimal effort.

To get the model into float16 format:
```
python export.py --weights yolov5s.pt --include onnx

# mo is installed with the openvino package.
mo --input_model yolov5s.onnx --data_type FP16
```

## TODOs
* Improve mapping SORT boxes back to detect boxes. Current method allows
double assignment, and this seems bugged.
* Add blob detector module. Kind of questionable b/c it is being phased out.
* Integrate `robotpy` for network tables support
* Support streaming the augmented camera feed to the driver station.
* Add stereo depth map if second camera is available.
* Add OpenVINO int8 quantization flow. Should accept the same dir hierarchy
  as the yolov5 training set since we need representative images during calibration.
* Support fp16 and int8 calibration for Jetson with TensorRT, and validate pytorch.
* Investigate async so that apriltags can update the feed more
  often than the YOLO detector
* Integrate some kind of OCR for bumper text detection

### Style
This project uses `black` because it's easy.
