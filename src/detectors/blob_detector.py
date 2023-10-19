import numpy as np
import cv2


# This...exists. But it's not terribly useful in any lighting conditions
# except what it's tuned for. At best this detector plugin is a starting point
# for implementing retroreflective target tracking.
class BlobDetector:
    # Colors is a vector of H(SV) values, not BGR triples!
    def __init__(self, colors=np.array((0))):
        self.colors = colors

    # The detect algorithm is...
    # - convert to HSV
    # - Recolor every pixel to the closest one from the database
    # - Extract contours from the segmented image
    # In OpenCV H goes from 0->180
    # https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
    def detect(self, frame):
        frame = self.generate_masks(frame)
        contour_list = self.contours(frame, min_area=400)

        dets = []
        for contour in contour_list:
            rect = cv2.minAreaRect(contour)
            corners = cv2.boxPoints(rect)
            dets.append(
                {
                    "type": "Blob",
                    "id": "contour",
                    "color": (255, 0, 255),
                    "confidence": 1,
                    "corners": corners,
                }
            )

        return dets

    # Generate binary masks for each color
    def generate_masks(self, frame):
        assert frame.shape[2] == 3
        hsvim = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        width = 20

        masks = np.empty((self.colors.shape[0], hsvim.shape[0], hsvim.shape[1]))
        for idx in range(self.colors.shape[0]):
            lowerb = np.array((self.colors[idx] - width, 0, 0))
            upperb = np.array((self.colors[idx] + width, 255, 255))
            masks[
                idx,
                :,
                :,
            ] = cv2.inRange(hsvim, lowerb, upperb)

        return masks.astype(np.uint8)

    def contours(self, masks, min_area=15):
        acc = []
        for idx in range(masks.shape[0]):  # 0 is the color panel
            contour_list, _ = cv2.findContours(
                masks[idx, ...], mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
            )
            filt = [x for x in contour_list if (cv2.contourArea(x) >= min_area)]
            acc = acc + filt
        return acc
