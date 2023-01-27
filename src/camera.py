import cv2
import subprocess

# I recommend that you set the camera to the highest frame rate that yields
# an acceptable exposure.
#
# For instance...
# v4l2-ctl -d 3 -c exposure_time_absolute=166
#
# Use v4l2-ctl --all -d <devicenum> to find the appropriate control
# There's probably also an auto exposure mode you need to disable.
# v4l2-ctl -d 3 -c auto_exposure=1 on the C310
def config_gain_exposure(devnum, gain, exp_time):
    tmp = {"exposure_auto": 1}
    if gain is not None:
        tmp["gain"] = gain

    if exp_time is not None:
        tmp["exposure_absolute"] = exp_time

    config_camera(devnum, tmp)


def config_camera(devnum, opt):
    cmd_list = ["v4l2-ctl"]
    cmd_list += ["-d", "{}".format(devnum)]
    for key in opt.keys():
        val = "{}={}".format(key, opt[key])
        cmd_list += ["-c", val]

    subprocess.call(cmd_list)


class CameraCtl:
    def __init__(
        self, devnum, dimensions=(480, 640), fps=30, gain=None, exposure=None, cal=None
    ):
        self.devnum = devnum
        self.shape = dimensions
        self.fps = fps

        self.gain = gain
        self.exposure = exposure
        self.cal = cal

        self.cap = cv2.VideoCapture(devnum, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(dimensions[0]))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(dimensions[1]))
        self.cap.set(cv2.CAP_PROP_FPS, int(fps))

        if not self.cap.isOpened():
            raise Exception("Could not open source. Try change the device ID.")

        # First read turns the source "on", this seems necessary for V4L2 commands
        # to stick
        _, _ = self.cap.read()

        if (gain is not None) or (exposure is not None):
            config_gain_exposure(self.devnum, self.gain, self.exposure)

    def __del__(self):
        self.cap.release()

    # Facilitates access to theOpenCV VideoCapture objects
    def get(self):
        return self.cap

    def read(self):
        err = self.cap.grab()
        if not err:
            return err, None

        err, frame = self.cap.retrieve()

        return err, frame
