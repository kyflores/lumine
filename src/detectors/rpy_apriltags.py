# Robotpy apriltags detector.
# Should be treated as the gold standard b/c it's based on
# what WPILib ships to teams.
import robotpy_apriltag as rtag
import cv2
import numpy as np

# Default parameter set to use for the Logitech C310
C310_PARAMS = (
    995.5027920759295,
    1001.3658254510876,
    618.5636884544525,
    369.80679933903093,
)


class RobotpyAprilTagDetector:
    def __init__(self, camera_params, tag_family, tag_size=1.0, dm_ceiling=70):
        self.camera_params = camera_params
        self.tag_size = tag_size
        self.tag_family = tag_family
        self.dm_ceiling = dm_ceiling

        cfg = rtag.AprilTagDetector.Config()
        cfg.debug = False
        cfg.decodeSharpening = 0.25
        cfg.numThreads = 4
        cfg.quadDecimate = 2.0
        cfg.quadSigma = 0.0
        cfg.refineEdges = True
        self.cfg = cfg

        self.posecfg = rtag.AprilTagPoseEstimator.Config(
            tagSize=tag_size,
            fx=camera_params[0],
            fy=camera_params[1],
            cx=camera_params[2],
            cy=camera_params[3],
        )

        self.det = rtag.AprilTagDetector()
        self.det.setConfig(self.cfg)
        self.det.addFamily("tag16h5", 0)

        self.pose = rtag.AprilTagPoseEstimator(self.posecfg)

    def get_family(self):
        return self.tag_family

    def detect(self, img):
        frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.det.detect(frame_gray)

        res = []
        for d in dets:
            dm = d.getDecisionMargin()
            scaled_dm = np.clip(dm, 0.0, self.dm_ceiling) / self.dm_ceiling

            tag_id = d.getId()
            center = d.getCenter()
            corners = np.array(d.getCorners([0] * 8)).reshape((4, 2)).astype(np.int)

            pose = self.pose.estimate(d)
            translation = pose.translation()
            tr_vec = translation.X(), translation.Y(), translation.Z()

            rotation = pose.rotation()
            ro_vec = rotation.X(), rotation.Y(), rotation.Z()

            res.append(
                {
                    "type": "apriltags",
                    "id": tag_id,
                    "color": (255, 0, 0),
                    "center": center,
                    "corners": corners,  # Pixel units
                    "confidence": scaled_dm,
                    "translation": tr_vec,  # tag_size units
                    "rotation_euler": ro_vec,  # degrees
                }
            )
        return res
