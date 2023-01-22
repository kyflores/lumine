# Installation script for virtualenvs.
# Might not work in conda if you've installed something with conda
# that brings along a different libstdc++ version

# virtualenv lumine-venv

# Install pytorch. See https://pytorch.org/ in the `Install Pytorch`
# section for instructions if not using the standard CUDA latest build
# (such as CPU only, ROCM, or Apple MPS on M1)
pip install torch torchvision

# Intel OpenVINO for AVX512 or IGP inference
pip install openvino
pip install openvino-dev # needed for `mo`

export MAKEFLAGS="-j$(nproc)"
pip install \
    numpy \
    matplotlib \
    scipy \
    opencv-python \
    prettytable \
    scikit-image \
    filterpy

# This has to come after numpy b/c it uses numpy as a build
# dependency
pip install lap

# robotpy's libraries
pip install pynetworktables
pip install "robotpy[all]"

