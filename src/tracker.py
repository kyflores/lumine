import numpy as np
import torch
import cv2
import torchvision.ops as ops
import subprojects.sort.sort as sort

# Corners and confidence to SORT (1,5) format.
# SORT wants [x,y,x,y,conf]
def cc2sort(corners, confidence):
    assert corners.shape == (4, 2)
    (x0, y0, w, h) = cv2.boundingRect(corners.astype(int))
    sort_xyxy = np.array((x0, y0, x0 + w, y0 + h, confidence))
    return sort_xyxy


class Sort:
    def __init__(self, max_age, min_hits, iou_threshold):
        self.sort_tracker = sort.Sort(
            max_age,
            min_hits,
            iou_threshold,
        )

        self.trackers = None

    def map_to_sort_id(self, dets, trackers):
        xyxys = [torch.tensor(cc2sort(x["corners"], 1)[:4]) for x in dets]

        # Collect all the detections into an (N, 4)
        det_c = torch.stack(xyxys)
        # Get track into an (M, 4)
        trk_c = torch.from_numpy(trackers[..., :4])

        ious = ops.box_iou(det_c, trk_c)

        # N (detections) is on rows, M (sort boxes) on columns.
        # dim=1 reduces horizontally, giving the max column index for each row
        # The index of the max IOU is the index in the sort result that
        # best matches that particular index in the detection result.
        mins = torch.argmax(ious, dim=1)

        used_boxes = set()
        for ix, m in enumerate(mins):
            dets[ix]["sort_id"] = int(trackers[m][-1])
            used_boxes.update([m.item()])

        # For each tracker that wasn't assigned to a box by IOU matching,
        # Draw a ghost box to indicate an expected target.
        all_trackers = set(np.arange(trackers.shape[0]))
        ghost_boxes = all_trackers - used_boxes
        for box in ghost_boxes:
            xmin, ymin, xmax, ymax = trackers[box][:4]
            dets.append(
                {
                    "type": "sort_occluded",
                    "id": "none",
                    "color": (128, 128, 128),
                    "confidence": 1,
                    "sort_id": -9,
                    "corners": np.array(
                        [[xmin, ymin],
                         [xmin, ymax],
                         [xmax, ymax,],
                         [xmax, ymin]]
                    )
                }
            )


    def update(self, detects):
        sort_dets = [cc2sort(d["corners"], d["confidence"]) for d in detects]
        if len(sort_dets) == 0:
            sort_dets = np.empty((0, 5))
        else:
            sort_dets = np.stack(sort_dets)

        # Match mapping is a Nx2, like
        # [[0 1]
        #  [1 0]
        #  [2 2]]
        # Where pairs map input detection indicies to tracker output indicies.
        self.trackers, match_mapping = self.sort_tracker.update(sort_dets)

        # Mutates `detects`, we return it again for convenience.
        if (self.trackers.shape[0] > 0) and (len(detects) > 0):
            # Assign box ID according to best IOU
            self.map_to_sort_id(detects, self.trackers)

        return detects
