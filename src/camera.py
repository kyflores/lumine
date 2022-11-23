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
    tmp = {"auto_exposure": 1, "gain": gain, "exposure_time_absolute": exp_time}
    config_camera(devnum, tmp)


def config_camera(devnum, opt):
    cmd_list = ["v4l2-ctl"]
    cmd_list += ["-d", "{}".format(devnum)]
    for key in opt.keys():
        val = "{}={}".format(key, opt[key])
        cmd_list += ["-c", val]

    subprocess.call(cmd_list)
