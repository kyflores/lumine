# Based on https://github.com/RangiLyu/nanodet/blob/main/demo_openvino/nanodet_openvino.cpp
import numpy as np
from scipy.special import softmax
import math
import cv2

# Used by all the pretrained nanodet variants.
STRIDES = (8, 16, 32, 64)
REG_MAX = 7
NMS_THRESH = 0.5


def into2xywh(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0]  # x origin
    y[:, 1] = x[:, 1]  # y origin
    y[:, 2] = x[:, 2] - x[:, 0]  # w
    y[:, 3] = x[:, 3] - x[:, 1]  # w
    return y


def generate_grid_center_priors(im_h, im_w):
    grids = []
    for s in STRIDES:
        feat_h = math.ceil(im_h / s)
        feat_w = math.ceil(im_w / s)

        # (2, feat_h, feat_w)
        hw_grid = np.mgrid[0:feat_h, 0:feat_w].astype(np.uint32)
        s_grid = np.ones((1, feat_h, feat_w), dtype=np.uint32) * s

        # (3, feat_h, feat_w)
        hws_grid = np.concatenate((hw_grid, s_grid), axis=0)
        hws_grid = hws_grid.reshape((3, feat_h * feat_w))

        grids.append(hws_grid.transpose())

    out = np.concatenate(grids)
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# grid (3598, 3)
# pred (1, 3598, 112) for COCO 80 classes
# In the 112 dim, it appears to be classes first then boxes
# 112: [classes: 0-80][x_min: reg_max + 1][y_min reg_max + 1][x_max: reg_max + 1][y_max: reg_max + 1]
def decode_infer(im_h, im_w, preds, grid_priors, threshold):
    # Now (3598, 112)
    preds = preds.squeeze()
    num_points, num_channels = preds.shape
    # Infer the number of classes from the input tensor size
    num_classes = num_channels - ((REG_MAX + 1) * 4)

    assert num_points == grid_priors.shape[0]
    # preds[:, 0:num_channels] = sigmoid(preds[:, 0:num_channels])
    best_score = np.max(preds[:, 0:num_classes], axis=1)
    class_ids = np.argmax(preds[:, 0:num_classes], axis=1)

    raw_boxes = preds[:, num_classes:]

    # Select indicies of all the boxes above the class confidence threshold
    valid_box_idx = best_score > threshold

    # Slice all the relevant arrays to include only the above threshold boxes
    valid_grid = grid_priors[valid_box_idx]
    valid_score = best_score[valid_box_idx]
    valid_class_ids = class_ids[valid_box_idx]
    valid_boxes = raw_boxes[valid_box_idx, :]

    assert valid_score.shape[0] == valid_grid.shape[0]
    assert valid_score.shape == valid_class_ids.shape
    assert valid_score.shape[0] == valid_boxes.shape[0]

    box_ret = dis_pred_to_bbox(im_h, im_w, valid_grid, valid_boxes)
    box_ret = into2xywh(box_ret)

    nms_res = cv2.dnn.NMSBoxes(box_ret, valid_score, threshold, NMS_THRESH)
    return (nms_res, box_ret, valid_score, valid_class_ids)


def dis_pred_to_bbox(im_h, im_w, grid, boxes):
    num_points, _ = boxes.shape

    # Now (3598, 4, 8)
    # The 4 dim are XYXY box values
    # The 8 I'm not sure, some kind of distribution since we need to take the softmax
    boxes_dist = boxes.reshape(num_points, 4, (REG_MAX + 1))
    after_softmax = softmax(boxes_dist, axis=2)
    z = np.arange(REG_MAX + 1)
    dis = (after_softmax * z).sum(axis=2)
    stride = grid[:, 2:]
    dis = stride * dis
    # dis is now 3598, 4

    feat_h = grid[:, 0:1]
    feat_w = grid[:, 1:2]

    # This weird slice on dis is to preserve a singleton dimension
    xmin = feat_w * stride - dis[:, 0:1]
    ymin = feat_h * stride - dis[:, 1:2]
    xmax = feat_w * stride + dis[:, 2:3]
    ymax = feat_h * stride + dis[:, 3:]
    out = np.stack([xmin, ymin, xmax, ymax])
    out = out.squeeze(-1).transpose()
    return out


if __name__ == "__main__":
    preds = np.random.randn(1, 3598, 112)
    grid = generate_grid_center_priors(416, 416)
    nms = decode_infer(416, 416, preds, grid, 2.0, 80)
