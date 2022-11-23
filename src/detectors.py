# Wrappers for different detector methods
import os

import torch
import numpy as np
import cv2

import prettytable as pt
import dt_apriltags as dtap
from scipy.spatial.transform import Rotation

# Default parameter set to use for the Logitech C310
C310_PARAMS = (
    995.5027920759295,
    1001.3658254510876,
    618.5636884544525,
    369.80679933903093,
)

# Relative to main.py
YOLOV5_PATH = os.path.dirname(__file__) + "/subprojects/yolov5/"
# Classes in the yolov5 model. Indexes should match the ultralytics/yolov5 pretrained models.
YOLOV5_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def detections_as_table(detections):
    table = pt.PrettyTable()
    table.field_names = ["type", "what", "sort_id", "xyxy"]
    for d in detections:
        table.add_row(
            [d["type"], d["id"], d.get("sort_id", -1), d["sort_xyxy"][:4].astype(int)]
        )

    return table.get_string()


class AprilTagDetector:
    def __init__(self, camera_params, tag_family, tag_size=1.0):
        self.camera_params = camera_params
        self.tag_size = tag_size
        self.tag_family = tag_family
        self.det = dtap.Detector(
            families=tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def get_family(self):
        return self.tag_family

    def detect(self, img):
        detections = self.det.detect(
            img,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        res = []
        for det in detections:
            ro = Rotation.from_matrix(det.pose_R).as_euler("zxy", degrees=True)
            corners_int = det.corners.astype(np.float32)
            (x0, y0, w, h) = cv2.boundingRect(corners_int)
            sort_xyxy = np.array((x0, y0, x0 + w, y0 + h, 1))
            res.append(
                {
                    "type": "apriltags",
                    "id": det.tag_id,
                    "color": (255, 0, 0),
                    "corners": corners_int,  # Pixel units
                    "sort_xyxy": sort_xyxy,
                    "translation": det.pose_t,  # tag_size units
                    "rotation_euler": ro,  # degrees
                }
            )

        return res


class YoloV5TorchDetector:
    def __init__(self, weights="yolov5s.pt", classes=YOLOV5_CLASSES):
        self.classes = classes
        self.det = torch.hub.load(YOLOV5_PATH, "custom", weights, source="local")

    def detect(self, img):
        detects = self.det.forward(img).xyxy[0].cpu().numpy()
        res = []
        for d in detects:
            # Order matters b/c of the script that draws them.
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))
            conf = d[4]
            classnm = self.classes[int(d[5])]
            res.append(
                {
                    "type": "yolov5",
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners,
                    "confidence": conf,
                    "sort_xyxy": np.append(
                        d[0:4], conf
                    ),  # Append causes realloc but w/e
                }
            )
        return res
