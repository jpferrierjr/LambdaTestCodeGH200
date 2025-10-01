#!/bin/bash

set -e  # Exit on any error

#--- Constants/Variables
#region variable initialization
CUDA_PATH="/usr/local/cuda"

PROFILE_FILE="${HOME}/.bashrc"

#VDW_BUILD_DIR="/tmp/vdw_build_$$"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
VENV_NAME=".venv"
VENV_PATH="${SCRIPT_DIR}/${VENV_NAME}"  

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
#endregion

# Activate source just in case
source ~/.bashrc

# Append WSL path
export PATH=$PATH:/usr/lib/wsl/lib

#--- Functions
#region functions used for systematic checks/installs

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}
#endregion


sudo apt-get install -y build-essential gfortran autoconf libtool pkg-config cmake curl wget tar


echo "Updating package lists..."
apt-get update
echo "Installing OpenBLAS and OpenMPI..."
apt-get install -y libopenblas-dev libopenmpi-dev

echo "🎉 OpenBLAS and OpenMPI installation complete."



#--- XC/FFTW3 references
#region XC & FFTW3 references install
sudo apt-get install -y libxc-dev libfftw3-mpi-dev

#endregion


#--- LIBVDWXC references
#region LIBVDWXC references install

# Install LibvdWXC
# print_status "Installing LibvdWXC..."
# rm -rf "${VDW_BUILD_DIR}"
# git clone https://gitlab.com/libvdwxc/libvdwxc.git "${VDW_BUILD_DIR}"
# cd "${VDW_BUILD_DIR}"

# git clean -fdx


# CPPFLAGS="-I/usr/local/fftw3/include"
# LDFLAGS="-L/usr/local/fftw3/lib"

# autoreconf -i

# ./configure --prefix=/usr/local/libvdwxc \
#             --enable-shared \
#             --enable-static

# make -j8
# sudo make install
# echo "🧹 Cleaning up..."
# cd "${SCRIPT_DIR}"
# rm -rf "${VDW_BUILD_DIR}"

#endregion



#--- Python
#region Python + Packages install

sudo apt-get install -y python3-dev python3-venv

# Create the virtual environment
python3 -m venv "${VENV_PATH}"

# Activate the virtual environment 
source "${VENV_PATH}/bin/activate"

# Upgrade and install
pip install --upgrade pip
pip install numpy scipy ase wheel mpi4py dftd4 cupy

#endregion



#--- GPAW
#region GPAW install

# Start the siteconfig.py file
cat > siteconfig.py << EOF
libraries = []
library_dirs = []
include_dirs = []
extra_link_args = []
extra_compile_args = []
runtime_library_dirs = []
define_macros = []

mpi = True
scalapack = True
parallel_python_interpreter = True

library_dirs += ['/usr/lib/x86_64-linux-gnu']
libraries += ['stdc++','pthread', 'm', 'dl']

# FFTW3
fftw = True
libraries += ['fftw3']
library_dirs += ['/usr/local/fftw3/lib']
include_dirs += ['/usr/local/fftw3/include']

# LibXC
libxc = True
libraries += ['xc']
library_dirs += ['/usr/local/libxc/lib']
include_dirs += ['/usr/local/libxc/include']

# LibvdWXC
libvdwxc = True
libraries += ['vdwxc']
library_dirs += ['/usr/local/libvdwxc/lib']
include_dirs += ['/usr/local/libvdwxc/include']

compiler = 'mpicc'
mpicompiler = 'mpicc'
libraries += ['mpi', 'scalapack-openmpi', 'blacs-openmpi']
extra_compile_args = ['-qopenmp']
extra_link_args = ['-qopenmp']

include_dirs += ['/usr/include']
libraries += ['xc', 'fftw3']

use_cuda = True
gpu = True
define_macros += [('GPAW_CUDA', '1')]
libraries += ['cudart', 'cuda', 'cublas', 'cusolver']
library_dirs += ['${CUDA_PATH}/lib64']
include_dirs += ['${CUDA_PATH}/include')]
gpu_target = 'cuda'
gpu_compiler = 'nvcc'
gpu_compile_args = ['-O3', '-g', '-gencode', 'arch=compute_86,code=sm_86']
EOF

# siteconfig.py is built. Saving locally and copying instead of moving.
# This allows the user to make changes if there were mistakes.
cp siteconfig.py ~/.gpaw/siteconfig.py

# Run the install for gpaw
pip install gpaw

gpaw info
#endregion