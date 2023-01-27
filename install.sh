# Installation script for Lumine.
# This script assumes an Ubuntu 22.04 system.

sudo apt install -y \
    python3-virtualenv \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-opencv \
    python3-skimage \
    python3-torch \
    python3-torchvision \
    python3-prettytable \
    v4l-utils

# TODO would be required for appsrc streamer.
# sudo apt install -y \
#     python3-gst-1.0  \
#     python3-gi \
#     gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-ugly

virtualenv lumine-venv --system-site-packages
source lumine-venv/bin/activate

# Intel OpenVINO for AVX512 or IGP inference
export MAKEFLAGS="-j$(nproc)"
pip install openvino openvino-dev filterpy lap

# robotpy's libraries
pip install pynetworktables "robotpy[all]"

