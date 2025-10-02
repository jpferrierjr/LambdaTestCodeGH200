#!/bin/bash

set -e  # Exit on any error

#--- Constants/Variables
#region variable initialization
CUDA_PATH="/usr/local/cuda"

PROFILE_FILE="${HOME}/.bashrc"
INSTALL_PREFIX=${HOME}/opt/gpaw_h200
GPAW_LIBS_PREFIX=${INSTALL_PREFIX}

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

NPROC=$(nproc)

# Activate source just in case
source ~/.bashrc

#--- Functions
#region functions used for systematic checks/installs

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}
#endregion


sudo apt-get install -y build-essential gfortran autoconf libtool pkg-config cmake curl wget tar
sudo apt-get install libblas-dev liblapack-dev libscalapack-mpi-dev libscalapack-openmpi-dev

echo "Updating package lists..."
sudo apt-get update
echo "Installing OpenBLAS and OpenMPI..."
sudo apt-get install -y libopenblas-dev libopenmpi-dev

echo "🎉 OpenBLAS and OpenMPI installation complete."



#--- XC/FFTW3 references
#region XC & FFTW3 references install
sudo apt-get install -y libxc-dev libfftw3-mpi-dev

#endregion


export CC=mpicc
export CXX=mpic++
export FC=mpifort
export F77=mpif77
export F90=mpif90

mkdir -p ${INSTALL_PREFIX}/src
cd ${INSTALL_PREFIX}/src

# -- FFTW --
# GPAW can use FFTW. Compiling it with ARM-specific optimizations is beneficial.
echo "Downloading and compiling FFTW..."
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xzvf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --prefix=${GPAW_LIBS_PREFIX} --enable-mpi --enable-openmp --enable-shared --enable-neon
make -j${NPROC}
make install
cd ..

# -- MAGMA --
# Matrix Algebra on GPU and Multicore Architectures for GPU-accelerated linear algebra.
echo "Downloading and compiling MAGMA..."
wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz
tar -xzvf magma-2.9.0.tar.gz
cd magma-2.9.0
# Create a make.inc for NVIDIA HPC SDK on ARM (Hopper architecture for H200)
# We use nvc/nvfortran directly here, not MPI wrappers, as MAGMA itself is not an MPI library.
cp make.inc-examples/make.inc.nvcc make.inc
sed -i 's/GPU_TARGET = Pascal/GPU_TARGET = Hopper/' make.inc
sed -i 's/CC = gcc/CC = nvc/' make.inc
sed -i 's/CXX = g++/CXX = nvc++/' make.inc
sed -i 's/FORT = gfortran/FORT = nvfortran/' make.inc
echo "prefix = ${GPAW_LIBS_PREFIX}" >> make.inc
make -j${NPROC}
make install
cd ..

# -- ELPA --
# Eigensolver for Petaflop-Scale Applications.
echo "Downloading and compiling ELPA..."
wget https://elpa.mpcdf.mpg.de/elpa/elpa-2025.06.001.tar.gz
tar -xzvf elpa-2025.06.001.tar.gz
cd elpa-2025.06.001
./configure --prefix=${GPAW_LIBS_PREFIX} CC=mpicc FC=mpifort CXX=mpicxx \
            --disable-sse-assembly --enable-gpu-nvidia-cuda CUDA_HOME=${CUDA_PATH}
make -j${NPROC}
make install
cd ..

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
pip install numpy scipy ase wheel mpi4py dftd4 cupy-12x

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

library_dirs += ['/usr/lib/aarch64-linux-gnu']
libraries += ['stdc++','pthread', 'm', 'dl']

library_dirs += ['/usr/lib']
include_dirs += ['/usr/include']

# FFTW3
fftw = True
libraries += ['fftw3']

# LibXC
libxc = True
libraries += ['xc']

# ELPA configuration
elpa = True
libraries += ['elpa_openmp']
library_dirs += ['/usr/local/elpa/lib']
include_dirs += ['${GPAW_LIBS_PREFIX}/include/elpa-2025.06.001/modules']

# LibvdWXC
# libvdwxc = True
# libraries += ['vdwxc']

# MAGMA
magma = True
if magma:
    libraries += ['cudart', 'cuda', 'cublas', 'cusolver', 'magma']
    library_dirs += ['${CUDA_PATH}/lib64', '/usr/local/magma/lib']
    include_dirs += ['${CUDA_PATH}/include', '/usr/local/magma/include']

compiler = 'mpicc'
mpicompiler = 'mpicc'
mpilinker = 'mpifort'

libraries += ['scalapack-openmpi', 'blas' ]

mpi = True
if mpi:
    libraries += ['mpi']

extra_compile_args += ['-O3', '-march=native', '-mtune=native', '-fopenmp']
extra_link_args = ['-fopenmp']

use_cuda = True
gpu = True

define_macros += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
if gpu:
    define_macros += [('GPAW_CUDA', '1')]
    libraries += ['cudart', 'cuda', 'cublas', 'cusolver', 'cufft']
    library_dirs += ['${CUDA_PATH}/lib64']
    include_dirs += ['${CUDA_PATH}/include')]
    gpu_target = 'cuda'
    gpu_compiler = 'nvcc'
    gpu_compile_args = ['-O3', '-g', '-gencode', 'arch=compute_90,code=sm_90']
EOF

# siteconfig.py is built. Saving locally and copying instead of moving.
# This allows the user to make changes if there were mistakes.
cp siteconfig.py ~/.gpaw/siteconfig.py

# Run the install for gpaw
LDFLAGS="-L/usr/lib/aarch64-linux-gnu" pip install gpaw

gpaw info
#endregion