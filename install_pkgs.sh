# Tested for py3.9
# Need build-essential and python3-dev for py3.10

# Installs the CPU oriented pytorch
# Use cu113, cu116, rocm5.1.1 instead for different platforms
pip install jupyterlab matplotlib opencv-python numpy scipy pupil-apriltags
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
