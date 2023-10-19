# Common functions for YoloV5
import numpy as np
import cv2
import os

# TODO: For OpenCV DNN
BOX_THRESH = 0.25
CLASS_THRESH = 0.25
NMS_THRESH = 0.45
NMS_SCORE_THRESH = BOX_THRESH * CLASS_THRESH

# Relative to main.py
YOLOV5_PATH = os.path.dirname(__file__) + "/../subprojects/yolov5/"
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


# owh is Origin, Width, Height. OpenCV uses this format
def xywh2owh(x):
    x[:, 0] = x[:, 0] - x[:, 2] / 2  # x origin
    x[:, 1] = x[:, 1] - x[:, 3] / 2  # y origin
    return x


# Squeeze the image into 640x640. If the image is nonsquare, the image is padded
# to 640^2 such that the corner near the origin is always filled.
def resize_to_frame(imraw, dim=640):
    major_dim = np.max(imraw.shape)
    scale = dim / major_dim
    outscale = 1 / scale
    imraw = cv2.resize(imraw, None, fx=scale, fy=scale)
    img = np.empty((dim, dim, 3), dtype=imraw.dtype)
    img.fill(114)
    img[: imraw.shape[0], : imraw.shape[1], :] = imraw
    return img, outscale


def process_yolo_output_tensor(tensor):
    tensor = tensor.squeeze()

    best_score = np.max(tensor[:, 5:], axis=1)
    # Where the best score >= the class thresh
    valid = best_score >= CLASS_THRESH
    tensor = tensor[valid]

    class_ids = np.argmax(tensor[:, 5:], axis=1)
    boxes = xywh2owh(tensor[:, :4])
    confidences = tensor[:, 4:5].squeeze() * best_score[valid]

    nms_res = cv2.dnn.NMSBoxes(boxes, confidences, NMS_SCORE_THRESH, NMS_THRESH)

    return (
        nms_res,
        boxes,
        confidences,
        class_ids,
    )


# From ultrlytics/yolov5/utils/augmentations
def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


# Calculates the position of a box on the original image using the ratio
# and (dw, dh) values from letterbox()
def unletterbox(xyxy, ratio, dwh):
    ratio = ratio[0]  # letterbox outputs (ratio, ratio) for some reason, just take on.
    (dw, dh) = dwh
