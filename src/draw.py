import cv2
import numpy as np


def draw(img, detections):
    # TODO placeholder skip if empty
    if len(detections) == 0:
        return img

    for d in detections:
        corners = [(d["corners"].clip(0, None).astype(int).reshape(4, 1, 2))]

        # Kind of misusing this, but I want boxes to all have different colors
        withbox = cv2.polylines(
            img, corners, isClosed=True, color=d["color"], thickness=3
        )

        newid = d.get("sort_id", -1)
        cv2.putText(
            withbox,
            "ID: {}".format(newid),
            d["corners"][0].astype(int),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )

    return withbox


# Debugging function for drawing sort boxes directly
def draw_sort(img, sorts):
    if sorts.shape[0] == 0:
        return img

    for box in sorts:
        x0, y0, x1, y1 = box[:4]
        classid = box[4]

        corners = np.empty((4, 1, 2), dtype=int)
        corners[0, :, :] = [x0, y0]
        corners[1, :, :] = [x0, y1]
        corners[2, :, :] = [x1, y1]
        corners[3, :, :] = [x1, y0]

        withbox = cv2.polylines(
            img, [corners], isClosed=True, color=(0, 0, 255), thickness=3
        )

        cv2.putText(
            withbox,
            "ID: {}".format(classid),
            np.array((x0, y0)).astype(int),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )
        return withbox
