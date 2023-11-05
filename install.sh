# Installation script for Lumine.
# This script assumes an Ubuntu 22.04 system.

sudo apt install -y python3-virtualenv v4l-utils
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

cd ${LUMINE_ROOT}
virtualenv lumine-venv
source lumine-venv/bin/activate

IS_VENV=$(python -c 'import sys; print(sys.prefix != sys.base_prefix)')
echo "In venv? ${IS_VENV}"
if [ "${IS_VENV}" = "False" ]; then
    echo "Detected that we are not in a venv. Something went wrong."
    exit 1
fi
echo "Created venv"

export MAKEFLAGS="-j$(nproc)"
# pip install numpy  # numpy as to already be installed when trying to install lap.
pip install -r requirements_yolov8.txt

# Build TVM
bash utils/tvmbuild.sh
pip install tvm/python/dist/tvm-*.whl

# Add the service file.
python utils/generate_systemd_unit.py
#sudo cp lumine.service /etc/systemd/system/
#sudo systemctl daemon-reload

echo "===================================================================="
echo ""
echo "Installation complete! Configure lumine by edting config/lumine.json"
echo "Use sudo \"systemctl enable lumine.service\" to start at boot."
echo ""
echo "===================================================================="
