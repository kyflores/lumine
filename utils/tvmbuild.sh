# First create a virtualenv and activate it before starting this script.

sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev python3-virtualenv
sudo apt-get install -y libvulkan-dev spirv-tools spirv-headers llvm-dev clinfo

# https://tvm.apache.org/docs/install/index.html
git clone --recursive https://github.com/apache/tvm --branch v0.14.0
pushd tvm

mkdir build
pushd build
cmake \
    -GNinja \
    -DUSE_VULKAN=ON \
    -DUSE_LLVM=ON \
    -DUSE_OPENCL=ON \
    ..
ninja
popd

# Go to the python/ dir
pushd python
python setup.py bdist_wheel
popd
