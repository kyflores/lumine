# First create a virtualenv and activate it before starting this script.

sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
sudo apt-get install -y python3-virtualenv libvulkan-dev spirv-tools spirv-headers llvm-dev

cd tvm
# https://tvm.apache.org/docs/install/index.html
git clone --recursive https://github.com/apache/tvm

mkdir build
cd build
cmake \
    -GNinja \
    -DUSE_VULKAN=ON \
    -DUSE_LLVM=ON \
    ..
ninja

# Go to the python/ dir
cd ../python
python setup.py bdist_wheel
