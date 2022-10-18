# API Basics
These are some starter examples to familiarize us with operations
on numpy arrays, which are the basic datatype used for both input and
output of python opencv.

# Environment
This assumes you have conda installed. Here is how to get a new environment
with the required packages. The astute reader might recognize that we're
using pip packages rather than python packages; there's a conflict between
torchvision and opencv (and possibly others) in the conda packages, so this
is a workaround.
```
conda create --name cv python=3.9 pip
conda activate cv


# Install the CPU only version for compatibility reasons.
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python matplotlib numpy scipy jupyterlab
```

If you make a mistake, simply remove the "cv" environment and start from the top.
```
conda deactivate cv
conda env remove -n cv
```

# Jupyter
We'll use juypter notebooks quite a bit. To start jupyter...
```
# First activate the environment
conda activate cv

# Start jupyter
jupyter lab
```

When jupyter is started this way it is accessible on the host by visiting
`localhost:8888/lab` in a web browser. It cannot be accessed over the network
when started this way.
