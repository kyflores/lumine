# Installation script for Lumine.
# This script assumes an Ubuntu 22.04 system.

sudo apt install -y \
    python3-virtualenv \
    python3-numpy \
    python3-scipy \
    python3-torch \
    python3-torchvision \
    python3-prettytable \
    v4l-utils
    # qt5dxcb-plugin
    # python3-opencv

# TODO would be required for appsrc streamer.
# sudo apt install -y \
#     python3-gst-1.0  \
#     python3-gi \
#     gstreamer1.0-plugins-good \
#     gstreamer1.0-plugins-bad \
#     gstreamer1.0-plugins-ugly

LUMINE_ROOT=$(dirname ${0})

virtualenv lumine-venv --system-site-packages
source ${LUMINE_ROOT}/lumine-venv/bin/activate


export MAKEFLAGS="-j$(nproc)"
pip install openvino openvino-dev filterpy lap black

# robotpy's libraries
pip install pynetworktables "robotpy[all]"

# Add the service file.
python ${LUMINE_ROOT}/utils/generate_systemd_unit.py
sudo cp ${LUMINE_ROOT}/lumine.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "===================================================================="
echo ""
echo "Installation complete! Configure lumine by edting config/lumine.conf"
echo "Use sudo \"systemctl enable lumine.service\" to start at boot."
echo ""
echo "===================================================================="
