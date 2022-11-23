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

# TODO: For OpenCV DNN
BOX_THRESH = 0.25
CLASS_THRESH = 0.25
NMS_THRESH = 0.45
NMS_SCORE_THRESH = BOX_THRESH * CLASS_THRESH

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


# Uses OpenCV dnn for inference. The model must be exported
# to ONNX for this backend.
# Also, the additional inference backends must be enabled in OpenCV
# at compile time. IIRC opencv-python build on PyPi does NOT enable
# cuda, vulkan, or openvino.
class YoloV5OpenCVDetector:
    def __init__(self, weights="yolov5s.onnx", classes=YOLOV5_CLASSES, backend="cpu"):
        self.classes = classes
        self.net = cv2.dnn.readNet(weights)
        if backend == "vulkan":
            print("Vulkan will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_VKCOM)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_VULKAN)
        elif backend == "opencl":
            print("OpenCL will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
        elif backend == "cuda":
            print("CUDA will be used if it is available.")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        elif backend == "cpu":
            print("CPU backend will be used.")
        else:
            print("Unknown backend, CPU backend will be used.")

        self.out_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, img):
        img = self._resize_to_frame(img)
        blob = cv2.dnn.blobFromImage(
            img,
            1.0 / 255,
            size=(img.shape[1], img.shape[0]),
            mean=(0.0, 0.0, 0.0),
            swapRB=False,
            crop=False,
        )

        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        (nms_res, boxes, confidences, class_ids) = self._process_net_out(outs)
        res = []
        for idx in nms_res:
            conf = confidences[idx]
            classnm = class_ids[idx]
            x, y, w, h = np.clip(boxes[idx], 0, 640).astype(np.uint32)
            d = (x, y, x + w, y + h)  # xyxy format
            corners = np.array(((d[0], d[1]), (d[0], d[3]), (d[2], d[3]), (d[2], d[1])))

            res.append(
                {
                    "type": "yolov5",
                    "id": classnm,
                    "color": (0, 255, 0),
                    "corners": corners,
                    "confidence": conf,
                    "sort_xyxy": np.append(d[0:4], conf),
                }
            )
        return res

    def _process_net_out(self, tensor):
        # 25200 is the total number of anchorboxes in the model output.
        tns = np.array(tensor).reshape(25200, len(self.classes) + 5)
        class_ids = np.empty(tns.shape[0])
        confidences = np.empty(tns.shape[0])
        boxes = np.empty((tns.shape[0], 4))

        count = 0
        for pred in tns[:]:
            assert pred.shape[0] == 5 + len(self.classes)

            class_scores = pred[5:]
            score_idx = np.argmax(class_scores)
            best_score = class_scores[score_idx]

            if best_score >= CLASS_THRESH:
                x, y, w, h, c = pred[:5]
                left = x - 0.5 * w
                top = y - 0.5 * h

                rect = np.array((left, top, w, h))

                class_ids[count] = score_idx
                confidences[count] = best_score * c
                boxes[count] = rect
                count += 1

        nms_res = cv2.dnn.NMSBoxes(
            boxes[:count], confidences[:count], NMS_SCORE_THRESH, NMS_THRESH
        )
        return (nms_res, boxes[:count], confidences[:count], class_ids[:count])

    def _resize_to_frame(self, imraw):
        major_dim = np.max(imraw.shape)
        scale = 640 / major_dim
        imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
        img = np.zeros((640, 640, 3), dtype=imraw.dtype)
        img[: imraw.shape[0], : imraw.shape[1], :] = imraw
        return img
