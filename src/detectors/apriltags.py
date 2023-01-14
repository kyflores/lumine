import dt_apriltags as dtap
from scipy.spatial.transform import Rotation
import numpy as np
import cv2

# Default parameter set to use for the Logitech C310
C310_PARAMS = (
    995.5027920759295,
    1001.3658254510876,
    618.5636884544525,
    369.80679933903093,
)


class AprilTagDetector:
    def __init__(self, camera_params, tag_family, tag_size=1.0):
        self.camera_params = camera_params
        self.tag_size = tag_size
        self.tag_family = tag_family
        self.det = dtap.Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def get_family(self):
        return self.tag_family

    def detect(self, img):
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = self.det.detect(
            frame_gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.tag_size,
        )

        res = []
        for det in detections:
            ro = Rotation.from_matrix(det.pose_R).as_euler("zxy", degrees=True)
            corners_int = det.corners.astype(np.float32)
            (x0, y0, w, h) = cv2.boundingRect(corners_int)

            # Experimentally, high 60's is a great detect, but 70.0 isn't a specified
            # maximum hence the clip.
            scaled_dm = np.clip(det.decision_margin, 0.0, 70.0) / 70.0
            res.append(
                {
                    "type": "apriltags",
                    "id": det.tag_id,
                    "color": (255, 0, 0),
                    "corners": corners_int,  # Pixel units
                    "confidence": scaled_dm,
                    "translation": det.pose_t,  # tag_size units
                    "rotation_euler": ro,  # degrees
                }
            )

        return res
