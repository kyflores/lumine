# lumine
Prototype multi object detector and tracker suite for FRC

## Install
Provided you are on ubuntu 22.04, just use
```
bash install.sh
```
`install.sh` installs everything it can from apt, creates a venv that
uses system packages, then adds a few python packages that aren't in apt repos.

Try `conda install -c conda-forge gcc=12.1.0` if you get an error about GLIBCXX_3.4.30

## Example Launch
```
python lumine.py --weights /path/to/openvino/weights/folder --source 0 --table --stream 5800 --draw
```

`weights`: Path to a folder containing openvino .bin, .xml, .mapping files
`source`: A name that can be passed to cv2.VideoCapture. Numbers refer to  /dev/videoX devices
`table`: Draw an ASCII table of the detection results

## Embedded Hardware Support (goals)
* CPU, with pytorch and OpenVINO backends.
* Intel IGPs with OpenVINO

## Custom detectors
All detectors should support a `detect(img) -> list(dict)` function.

`detect` is passed a 3 channel RGB image as a numpy array of shape (H, W, 3).
Detectors should not modify the image they receive, if you must modify it,
create a copy and modify that.

Detectors should produce a dictionary with at least the following
for each detection instance.
* `type`, Which detector this originated from.
* `id` Could be a tag family, object type. Info about what's in the bounding box.
* `color` RGB triple for the box color. Used when drawing the feed.
* `corners` of the bounding area of the detection
  * numpy array, 4 rows, 2 columns, shape = (4, 2)
  * `corners` should be scaled to the input image: If the input is rescaled inside the
     `detect` function, corners must be scaled back to the original.
* `confidence` A confidence value between 0 and 1 for the detection.
  * 0 is worst, 1.0 is best. Scale the value if it's not between 0 and 1.0.

`detect` should always return a list of detections. If there is only one detection, it
should still be in a list.

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
v4l2-ctl -d 3 -c exposure_auto=1
v4l2-ctl -d 3 -c gain=200
v4l2-ctl -d 3 -c exposure_time_absolute=250
```

OpenCV can configure these parameters with the `cv2.CAP_PROP_*` fields,
but just from experimenting with it, it doesn't seem reliable. Setting
the gain for instance changes the number reported by `v4l2-ctl` but
doesn't affect the image.

## Networktables
If using `lumine` with network tables, pass `--nt TE.AM` on the command line or pass
a full ip like `127.0.0.1`.

Lumine creates the table `/lumine` and a subtable for each detector family. For instance,
`lumine/yolo` and `lumine/apriltags`.

Within each detector specific table, data is stored in a "struct of arrays" consisting
of an array of scalars for each property. For instance, `yolo` produces...
```
/lumine/yolo

len: <number>
ids: [0, 4, 62, 1....]
confidence: [0.5, 0.42, 0.32, 0.12....]
```
To access all the properties for a one particular detection, you must request the same
index from each property array.

Detections are always sorted in order of descending confidence.

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
python export.py --weights /path/to/your/weights.pt --include onnx --opset 12

# mo is installed with the openvino-dev package.
mo --input_model /path/to/your/weights.onnx --data_type FP16
```

OpenVINO produces a `.xml, .bin, and .mapping` file from your model, which should be put
in its own folder. Then, passing the path to the folder to `--weights` when calling lumine.

YoloV8 can be easily exported to OpenVINO weights by the Ultralytics command line util.
```
yolo export model=yolov8n.pt format=openvino half=true
```

## Retraining Models
Lumine aims to offer some choice of object detection model, depending on the coprocessor in use.
The YoloV5 or YoloV8 families will be the first choice for powerful coprocessors
that provide acceleration hardware like NVIDIA Jetson or Intel Iris Xe. More limited systems
can benefit from a lighter model like NanoDet.

We recommend `CVAT` (cvat.ai) for image labeling. CVAT is a labeling server developed by
Intel that runs a web interface that users can connect to and label images. CVAT is primarily
self-hosted, but the setup procedure is relatively low-effort thanks to its docker-compose
procedure. Some cloud based solutions are Roboflow or Supervisely, but may have certain limitations
in the free-tier.

### YOLO
Retraining YOLO is quite easy thanks to how complete the Ultralytics solution is.
Start by exporting to the YOLO 1.1 format in CVAT.
TODO

### Autolabeling
Lumine provides a script (`utils/autolabel.py`) that uses an existing model to generate YOLO format
labels to be imported into CVAT. The workflow would look something like this.
* Collect and label a small dataset. For FRC teams, this could be the field walkthrough videos that
  FIRST releases on day one.
* Train the largest instance of YOLO your hardware can handle on the small dataset.
* Collect additional datasets. Have autolabel analyze them and generate labels.
* Import labels into CVAT and manually correct errors. Add the finished labels to the cumulative
  dataset
* Retrain the model with the cumulative dataset and repeat the process. As the dataset improves
  autolabel should produce better results.

Ideally autolabeling can progress to a state where it's practical for teams to add data collected
from practice matches on the first day of competition, and produce a model with higher
performance in their particular environment as quickly as possible.

## Dev TODOs
* Improve mapping SORT boxes back to detect boxes. Current method allows
double assignment, and this seems bugged.
  * We might want to run a single sort instance per detector type
* Add blob detector module. Kind of questionable b/c it is being phased out.
* Support streaming the augmented camera feed to the driver station.
  * Test cscore module
* Add OpenVINO int8 quantization flow. Should accept the same dir hierarchy
  as the yolov5 training set since we need representative images during calibration.
  * int8 is only for CPUs, so maybe this is not useful. Xe IGPU cannot benefit from int8
* Support fp16 and int8 calibration for Jetson with TensorRT, and validate pytorch.
* Figure out how to request a faster update rate from NT
* Support multiple cameras and switching
* Add a timestamp to networktables API
* Support Nanodet training and inference.
* (High effort) Implement a Lumine vendor library that provides an RPC interface for
  querying detects rather than using network tables.
* Replace cscore with a gstreamer solution for HW accelerated temporal codecs.
  * Also need to provide some sort of driver station client or dashboard plugin

### Style
This project uses `black` because it's easy.
