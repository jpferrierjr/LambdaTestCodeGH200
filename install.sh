#!/bin/bash

set -e  # Exit on any error

#--- Constants/Variables
#region variable initialization
CUDA_PATH="/usr/local/cuda"

PROFILE_FILE="/home/ubuntu/.bashrc"
INSTALL_PREFIX="/home/ubuntu/opt/gpaw_h200"
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

# print_status() {
#     echo -e "${GREEN}[INFO]${NC} $1"
# }
# #endregion


sudo apt-get install -y build-essential gfortran autoconf libtool pkg-config cmake curl wget tar
sudo apt-get install libblas-dev liblapack-dev libscalapack-mpi-dev libscalapack-openmpi-dev

echo "Updating package lists..."
sudo apt-get update
echo "Installing OpenBLAS and OpenMPI..."
sudo apt-get install -y libopenblas-dev libopenmpi-dev libudev-dev

# echo "🎉 OpenBLAS and OpenMPI installation complete."



# #--- XC/FFTW3 references
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
if [ -d /usr/local/fftw ]; then
echo "FFTW3 directory already exists."
else
if [ -d ${INSTALL_PREFIX}/src/fftw-3.3.10 ]; then
cd fftw-3.3.10
else
echo "Downloading and compiling FFTW..."
wget https://www.fftw.org/fftw-3.3.10.tar.gz
tar -xzvf fftw-3.3.10.tar.gz
cd fftw-3.3.10
fi
./configure --prefix=/usr/local/fftw --enable-mpi --enable-openmp --enable-shared --enable-neon
make -j${NPROC}
make install
cd ${INSTALL_PREFIX}/src
fi

#-- MAGMA --
if [ -d /usr/local/magma ]; then
echo "MAGMA directory already exists."
else

if [ -d {INSTALL_PREFIX}/src/magma-2.9.0 ]; then
cd magma-2.9.0
else
wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.9.0.tar.gz
tar -zxvf magma-2.9.0.tar.gz
cd magma-2.9.0
# Use CMake build system which handles CUDA compatibility better
mkdir -p build
cd build
fi

# Configure with CMake for better CUDA 13.0 compatibility
cmake .. \
-DCMAKE_INSTALL_PREFIX=/usr/local/magma \
-DCMAKE_BUILD_TYPE=Release \
-DGPU_TARGET="Hopper" \
-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
-DUSE_CUDA=ON \
-DMAGMA_ENABLE_CUDA=ON \
-DBLA_VENDOR=OpenBLAS \
-DCMAKE_C_FLAGS="-O3 -DADD_ -fPIC" \
-DCMAKE_CXX_FLAGS="-O3 -DADD_ -fPIC -std=c++11" \
-DCMAKE_Fortran_FLAGS="-O3 -DADD_ -fPIC" \
-DOpenBLAS_ROOT=/usr/local/openblas

# Configure MAGMA make.inc for our setup
cat > make.inc << EOF
CC        = gcc
CXX       = g++
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O3 -DADD_ -Wall -Wno-unused-function -fPIC -fopenmp -mtune=native
F77OPTS   = -O3 -DADD_ -Wall -Wno-unused-dummy-argument -fPIC -fopenmp -mtune=native
FOPTS     = -O3 -DADD_ -Wall -x f95-cpp-input -fPIC -fopenmp -mtune=native
NVOPTS    = -O3 -DADD_ -Xcompiler -fPIC -Xcompiler "-DADD_"

GPU_TARGET = Hopper

LIB       = -lopenblas -lcudart -lcublas -lcusparse -lcusolver

CUDADIR   = $CUDA_PATH
OPENBLASDIR = /usr/share/doc/libopenblas-dev

LIBDIR    = -L\$(CUDADIR)/lib64 -L\$(OPENBLASDIR)/lib
INC       = -I\$(CUDADIR)/include -I\$(OPENBLASDIR)/include
DEVCCFLAGS = -std=c++14 -DADD_
EOF

make -j${NPROC}
sudo make install prefix=/usr/local/magma
cd ${INSTALL_PREFIX}/src

fi



# -- ELPA --
# Eigensolver for Petaflop-Scale Applications.
cd ${INSTALL_PREFIX}/src
if [ -d /usr/local/elpa ]; then
echo "ELPA directory already exists."
else
if [ -d ${INSTALL_PREFIX}/src/elpa-2025.06.001 ]; then
cd elpa-2025.06.001
else
echo "Downloading and compiling ELPA..."
wget https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/2025.06.001/elpa-2025.06.001.tar.gz
tar -xzvf elpa-2025.06.001.tar.gz
cd elpa-2025.06.001
fi
make distclean || true
mkdir -p build
cd build
../configure --prefix=/usr/local/elpa CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort CFLAGS="-O3 -mtune=native" CXXFLAGS="-O3 -mtune=native" FCFLAGS="-O3 -mtune=native" FFLAGS="-O3 -mtune=native" LIBS="-lstdc++ -lm" --enable-openmp --disable-sse-kernels --disable-avx-kernels --disable-avx2-kernels --disable-sse-assembly-kernels --disable-avx512-kernels --enable-neon-arch64-kernels --with-NVIDIA-GPU-compute-capability=sm_90 --enable-nvidia-gpu-kernels --with-cusolver=yes --with-cuda-path=${CUDA_PATH} --with-mpi=yes
make -j${NPROC}
make install
cd ${INSTALL_PREFIX}/src
fi

#--- LIBVDWXC references
# region LIBVDWXC references install

# Install LibvdWXC
# echo "Installing LibvdWXC..."
# cd ${INSTALL_PREFIX}/src
# git clone https://gitlab.com/libvdwxc/libvdwxc.git "${VDW_BUILD_DIR}"
# cd ${INSTALL_PREFIX}/src

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

cd /home/ubuntu/GH200/LambdaTestCodeGH200

# Create the virtual environment
python3 -m venv "${VENV_PATH}"

# Activate the virtual environment 
source "${VENV_PATH}/bin/activate"

# Upgrade and install
pip install --upgrade pip
pip install numpy==1.26.4 scipy ase wheel mpi4py dftd4 cupy-cuda12x

#endregion



#--- GPAW
#region GPAW install

# Extra precaution since some servers don't allow access to /root,
# yet they set ~/=/root. Dumb
export LD_LIBRARY_PATH=/usr/local/fftw/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/magma/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/elpa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=/usr/local/fftw/lib:$LIBRARY_PATH
export LIBRARY_PATH=/usr/local/magma/lib:$LIBRARY_PATH
export LIBRARY_PATH=/usr/local/elpa/lib:$LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

export C_INCLUDE_PATH=/usr/local/fftw/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/magma/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/elpa/include/elpa_openmp-2025.06.001/:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/elpa/include/elpa_openmp-2025.06.001/elpa/:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/elpa/include/elpa_openmp-2025.06.001/modules:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/elpa/include/elpa_openmp-2025.06.001/elpa/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH

export GPAW_CONFIG=/home/ubuntu/.gpaw/siteconfig.py

# Start the siteconfig.py file
cat > siteconfig.py << EOF
mpi         = True
scalapack   = True
fftw        = True
libxc       = True
elpa        = True
magma       = True
#libvdwxc   = True
use_cuda    = True
gpu         = True
parallel_python_interpreter = True

compiler    = 'mpicc'
mpicompiler = 'mpicc'

library_dirs    += ['/usr/lib/aarch64-linux-gnu']
library_dirs    += ['/usr/lib']
include_dirs    += ['/usr/include']

# ELPA configuration
if elpa:
    libraries       += ['elpa_openmp']
    library_dirs    += ['/usr/local/elpa/lib']
    include_dirs    += ['/usr/local/elpa/include/elpa_openmp-2025.06.001/modules', '/usr/local/elpa/include/elpa_openmp-2025.06.001/elpa/include', '/usr/local/elpa/include/elpa_openmp-2025.06.001/elpa/']

# Scalapack
if scalapack:
    define_macros   += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
    define_macros   += [('GPAW_FFTW_UNDERSCORE_BLACS', '1')]
    libraries       += [ 'scalapack-openmpi' ]

# MAGMA
if magma:
    libraries += ['cudart', 'cuda', 'cublas', 'cusolver', 'magma']
    library_dirs += ['/usr/local/magma/lib']
    include_dirs += ['/usr/local/magma/include']

# FFTW3
if fftw:
    libraries += ['fftw3']
    library_dirs += ['/usr/local/fftw/lib']
    include_dirs += ['/usr/local/fftw/include']

# LibXC
if libxc:
    libraries += ['xc']

libraries   += [ 'blas' ]

# MPI
if mpi:
    libraries           += ['mpi']
    extra_compile_args  += ['-O3', '-march=native', '-mtune=native', '-fopenmp']
    extra_link_args     = ['-fopenmp']

# LibvdWXC
if libvdwxc:
#   libraries += ['vdwxc']

# GPU
if gpu:
    define_macros   += [('GPAW_CUDA', '1')]
    libraries       += ['cudart', 'cuda', 'cublas', 'cusolver', 'cufft']
    library_dirs    += ['/usr/local/cuda/lib64']
    include_dirs    += ['/usr/local/cuda/include']
    gpu_target      = 'cuda'
    gpu_compiler    = 'nvcc'
    gpu_compile_args= ['-O3', '-g', '-gencode', 'arch=compute_90,code=sm_90']

libraries   += [ 'stdc++','pthread', 'm', 'dl', 'gfortran' ]
EOF

# siteconfig.py is built. Saving locally and copying instead of moving.
# This allows the user to make changes if there were mistakes.
mkdir -p /home/ubuntu/.gpaw
cp siteconfig.py /home/ubuntu/.gpaw/siteconfig.py

# Run the install for gpaw
#LDFLAGS="-L/usr/lib/aarch64-linux-gnu" pip install gpaw
pip install gpaw

gpaw info
#endregion