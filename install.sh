#!/bin/bash

#### NOTE:
# It's important to realize that this script is written specifically for an ARM64 server that already has CUDA installed.
# This will not work on any other type of server, since it is installing ARM specific softwares. You should modify this script
# if you plan on using it on another type of server.


set -e  # Exit on any error

#--- Constants/Variables
#region variable initialization

# This changes from device to device. So, Change it here.
# This is supposed to be ~/ but some servers don't allow that access
HOME_PATH="/home/ubuntu"

# Set the bash location and the install directory
BASH_FILE="${HOME_PATH}/.bashrc"
INSTALL_PATH="${HOME_PATH}/gpaw_install"

# Setup the path for the actual code (where our python code/environment will be)
CODE_PATH="${HOME_PATH}/code"

VENV_NAME=".venv"
VENV_PATH="${CODE_PATH}/${VENV_NAME}"  

ABS_LIB_PATH="/usr/lib/aarch64-linux-gnu"



# Set the library path

# Set the paths for each library
LIB_BASE="/usr/local"
GEN_INCL="/usr/include"

CUDA_PATH="$LIB_BASE/cuda"
ELPA_PATH="$LIB_BASE/elpa"
MAGMA_PATH="$LIB_BASE/magma"
VDW_PATH="$LIB_BASE/libvdwxc"
FFTW_PATH=$ABS_LIB_PATH


# Set the library name
FFTW_LIB_NAME="fftw3-mpi"
BLAS_LIB_NAME="openblas64"
MPI_LIB_NAME="mpi"
SCLPCK_LIB_NAME="scalapack-openmpi"
LPCK_LIB_NAME="lapack64"
LBXC_LIB_NAME="xc"

ELPA_LIB_NAME="elpa_openmp"
MAGMA_LIB_NAME="magma"

# Custom Paths
BLAS_PATH="$ABS_LIB_PATH/openblas64-openmp"

# Set the versions of each software
ELPA_VER="2023.11.001"
MAGMA_VER="2.9.0"

# Names of the softwares that have to be manually installed
ELPA_NM="elpa-$ELPA_VER"          # sudo apt-get install libelpa-dev if no GPU support. We want GPU support
MAGMA_NM="magma-$MAGMA_VER"       # Almost no repos. Outdated if exist. Need to manually install




# Set the number of processors for parallel Make
NPROC=$(nproc)

#endregion


# Activate source just in case
source ~/.bashrc

#--- Functions
#region functions used for systematic checks/installs
# update the libaries
sudo apt-get update -y

# Install anything necessary.
sudo apt-get install -y build-essential gfortran autoconf libtool pkg-config cmake curl wget tar g++ libstdc++-12-dev automake

# Install packages we won't explicitly compile
sudo apt-get install -y libopenmpi-dev libscalapack-mpi-dev libopenblas64-openmp-dev libudev-dev

#endregion


# Explicitly set the compilers
export CC=mpicc
export CXX=mpic++
export FC=mpif90
export F77=mpif77
export F90=mpif90
export CUDA_HOME=$CUDA_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDADIR=$CUDA_PATH
export OPENBLASDIR=$BLAS_PATH
export OMP_NUM_THREADS=${NPROC}

CPPFLAGS="-I$GEN_INCL"
LDFLAGS="-L$LIB_BASE"


# #--- XC/FFTW3 references
#region XC & FFTW3 references install

# -- FFTW -- //Dependent on OpenMPI, OpenMP (done above)
sudo apt-get install -y libfftw3-dev libfftw3-mpi-dev libxc-dev

#endregion

#--- LIBVDWXC references
#region LIBVDWXC references install

#--- Install LibvdWXC
#region
echo "Installing LibvdWXC..."
cd $INSTALL_PATH
git clone https://gitlab.com/libvdwxc/libvdwxc.git
cd libvdwxc

autoreconf -i

./configure --prefix=$VDW_PATH \
    CC=mpicc \
    --enable-shared \
    --enable-static \
    CFLAGS="-O3 -march=native -fPIC"

make -j${NPROC}
sudo make install
#endregion


# -- ELPA --
#region Eigensolver for Petaflop-Scale Applications.
cd ${INSTALL_PATH}
if [ -d ${ELPA_PATH} ]; then
    echo "ELPA directory already exists."
else
    if [ -d ${INSTALL_PATH}/${ELPA_NM} ]; then
        cd ${ELPA_NM}
    else
        echo "Downloading and compiling ELPA..."
        wget https://elpa.mpcdf.mpg.de/software/tarball-archive/Releases/${ELPA_VER}/${ELPA_NM}.tar.gz
        tar -xzvf ${ELPA_NM}.tar.gz
        cd ${ELPA_NM}
    fi

    # Check if ARM PL exist. If so, compile with it
    ./configure --prefix=${ELPA_PATH} \
    CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort \
    CFLAGS="-O3 -mtune=native -fPIC" \
    CXXFLAGS="-O3 -mtune=native -fPIC" \
    FCFLAGS="-O3 -mtune=native -fPIC" \
    FFLAGS="-O3 -mtune=native -fPIC" \
    LDFLAGS="-L$ABS_LIB_PATH" \
    LIBS="-lstdc++ -lm -l$BLAS_LIB_NAME -l$SCLPCK_LIB_NAME" \
    SCALAPACK_LDFLAGS="-l$SCLPCK_LIB_NAME" \
    --enable-openmp \
    --disable-sse-kernels \
    --disable-avx-kernels \
    --disable-avx2-kernels \
    --disable-sse-assembly-kernels \
    --disable-avx512-kernels \
    --enable-neon-arch64-kernels \
    --with-NVIDIA-GPU-compute-capability=sm_90 \
    --enable-nvidia-gpu-kernels \
    --with-cusolver=yes \
    --with-cuda-path=${CUDA_PATH} \
    --with-mpi=yes
    
    make -j${NPROC}
    make install
fi
ELPA_INCLUDE_DIR=$(find ${ELPA_PATH}/include -type d -name "elpa*" | head -n 1)
#endregion


#-- MAGMA --
#region
if [ -d ${MAGMA_PATH} ]; then
    echo "MAGMA directory already exists."
else

    if [ -d ${INSTALL_PATH}/${MAGMA_NM} ]; then
        cd $MAGMA_NM
    else
        wget https://icl.utk.edu/projectsfiles/magma/downloads/$MAGMA_NM.tar.gz
        tar -zxvf $MAGMA_NM.tar.gz
        cd $MAGMA_NM
    fi

    # Configure with CUDA compatibility. Server has GH200 = Hopper Target/sm_90
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=$MAGMA_PATH \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGET=sm_90 \
    -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH \
    -DUSE_CUDA=ON \
    -DMAGMA_ENABLE_CUDA=ON \
    -DBLA_VENDOR=OpenBLAS \
    -DLAPACK_LIBRARIES="-l$LPCK_LIB_NAME -l$BLAS_LIB_NAME" \
    -DBLAS_LIBRARIES="$ABS_LIB_PATH/libopenblas64.so" \
    -DLAPACK_LIBRARIES="$ABS_LIB_PATH/libopenblas64.so" \
    -DCMAKE_C_FLAGS="-O3 -DADD_ -fPIC -fopenmp" \
    -DCMAKE_CXX_FLAGS="-O3 -DADD_ -fPIC -fopenmp -std=c++14" \
    -DCMAKE_Fortran_FLAGS="-O3 -DADD_ -fPIC -fopenmp"

    make -j${NPROC}
    sudo make install prefix=$MAGMA_PATH
    cd ${INSTALL_PREFIX}/src

fi
#endregion






#--- Python
#region Python + Packages install

sudo apt-get install -y python3-dev python3-venv

cd $CODE_PATH

# Create the virtual environment
python3 -m venv "${VENV_NAME}"

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
export LD_LIBRARY_PATH=$MAGMA_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$ELPA_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=/usr/local/fftw/lib:$LIBRARY_PATH
export LIBRARY_PATH=$MAGMA_PATH/lib:$LIBRARY_PATH
export LIBRARY_PATH=$ELPA_PATH/lib:$LIBRARY_PATH
export LIBRARY_PATH=$CUDA_PATH/lib64:$LIBRARY_PATH

export C_INCLUDE_PATH=/usr/local/fftw/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$MAGMA_PATH/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$ELPA_PATH/include/elpa_openmp-2023.11.001/:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$ELPA_PATH/include/elpa_openmp-2023.11.001/elpa/:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$ELPA_PATH/include/elpa_openmp-2023.11.001/modules:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$ELPA_PATH/include/elpa_openmp-2023.11.001/elpa/include:$C_INCLUDE_PATH
export C_INCLUDE_PATH=$CUDA_PATH/include:$C_INCLUDE_PATH

export GPAW_CONFIG=$HOME_PATH/.gpaw/siteconfig.py

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

library_dirs    += ['${ABS_LIB_PATH}']
library_dirs    += ['/usr/lib']
include_dirs    += ['${GEN_INCL}']

# ELPA configuration
if elpa:
    libraries       += ['${ELPA_LIB_NAME}']
    library_dirs    += ['${ELPA_PATH}/lib']
    include_dirs    += ['${ELPA_INCLUDE_DIR}/modules', '${ELPA_INCLUDE_DIR}/elpa/include', '${ELPA_INCLUDE_DIR}/elpa']

# Scalapack
if scalapack:
    define_macros   += [('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
    define_macros   += [('GPAW_FFTW_UNDERSCORE_BLACS', '1')]
    libraries       += [ '${SCLPCK_LIB_NAME}' ]

# MAGMA
if magma:
    libraries += ['cudart', 'cuda', 'cublas', 'cusolver', '${MAGMA_LIB_NAME}']
    library_dirs += ['${MAGMA_PATH}/lib']
    include_dirs += ['${MAGMA_PATH}/include']
    define_macros += [('GPAW_WITH_MAGMA', '1')]

# FFTW3
if fftw:
    libraries += ['${FFTW_LIB_NAME}']
    library_dirs += ['${ABS_LIB_PATH}']
    include_dirs += ['${GEN_INCL}']

# LibXC
if libxc:
    libraries += ['${LBXC_LIB_NAME}']

libraries       += ['${BLAS_LIB_NAME}']
library_dirs    += ['${BLAS_PATH}']


# MPI
if mpi:
    libraries           += ['${MPI_LIB_NAME}']
    extra_compile_args  += ['-O3', '-march=native', '-mtune=native', '-fopenmp']
    extra_link_args     = ['-fopenmp']
    define_macros       += [('PARALLEL', '1')]

# LibvdWXC
if libvdwxc:
    libraries += ['vdwxc']
    library_dirs += ['${VDW_PATH}']

# GPU
if gpu:
    define_macros   += [('GPAW_CUDA', '1')]
    libraries       += ['cudart', 'cuda', 'cublas', 'cusolver', 'cufft']
    library_dirs    += ['${CUDA_PATH}/lib64']
    include_dirs    += ['${CUDA_PATH}/include']
    gpu_target      = 'cuda'
    gpu_compiler    = 'nvcc'
    gpu_compile_args= ['-O3', '-g', '-gencode', 'arch=compute_90,code=sm_90']

libraries   += [ 'stdc++','pthread', 'm', 'dl', 'gfortran' ]
EOF

# siteconfig.py is built. Saving locally and copying instead of moving.
# This allows the user to make changes if there were mistakes.
mkdir -p $HOME_PATH/.gpaw
cp siteconfig.py $HOME_PATH/.gpaw/siteconfig.py

# Run the install for gpaw
pip install --no-build-isolation --no-cache-dir gpaw

gpaw info
#endregion

# Explicitly add paths to bashrc
BASHRC="$HOME_PATH/.bashrc"
cat >> ${BASHRC} << EOF

# GPAW environment
export LD_LIBRARY_PATH=${ELPA_PATH}/lib:${MAGMA_PATH}/lib:${CUDA_PATH}/lib64:${VDW_PATH}/lib:\$LD_LIBRARY_PATH
export PATH=${CUDA_PATH}/bin:\$PATH
export OMP_NUM_THREADS=1
export GPAW_CONFIG=${HOME_PATH}/.gpaw/siteconfig.py

# Activate virtual environment
source ${VENV_PATH}/bin/activate
EOF

# Run source ~/.bashrc
source ~/.bashrc
source "${VENV_PATH}/bin/activate"