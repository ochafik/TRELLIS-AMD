#!/bin/bash
# Portable HIP build script for diff-gaussian-rasterization
# Works with any venv and ROCm installation

set -e

# Auto-detect paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HIP_RASTERIZER="${SCRIPT_DIR}/hip_rasterizer"
BUILD_DIR="${SCRIPT_DIR}/build_hip"

# Find Python version and venv
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_DIR="$VIRTUAL_ENV"
else
    echo "ERROR: No virtual environment active. Please activate your venv first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_SUFFIX="cpython-${PYTHON_VERSION/./}-x86_64-linux-gnu"

# Find ROCm
if [ -d "/opt/rocm" ]; then
    ROCM_DIR="/opt/rocm"
elif [ -d "/opt/rocm-6.4.2" ]; then
    ROCM_DIR="/opt/rocm-6.4.2"
else
    # Find any rocm installation
    ROCM_DIR=$(ls -d /opt/rocm* 2>/dev/null | head -1)
    if [ -z "$ROCM_DIR" ]; then
        echo "ERROR: ROCm not found in /opt/rocm*"
        exit 1
    fi
fi

# Find GPU architecture
GPU_ARCH=$(rocminfo | grep -o 'gfx[0-9a-z]*' | head -1)
if [ -z "$GPU_ARCH" ]; then
    echo "WARNING: Could not detect GPU architecture, defaulting to gfx1100"
    GPU_ARCH="gfx1100"
fi

echo "=== Building diff-gaussian-rasterization for AMD GPU ==="
echo "VENV: $VENV_DIR"
echo "ROCm: $ROCM_DIR"
echo "GPU: $GPU_ARCH"
echo "Python: $PYTHON_VERSION"

# PyTorch paths
TORCH_INCLUDE="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/torch/include"
TORCH_LIB="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/torch/lib"

# Verify paths exist
if [ ! -d "$TORCH_INCLUDE" ]; then
    echo "ERROR: PyTorch include directory not found: $TORCH_INCLUDE"
    echo "Make sure PyTorch is installed in your venv"
    exit 1
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Compile flags
HIPCC="${ROCM_DIR}/bin/hipcc"
COMMON_FLAGS="-fPIC -O3 -std=c++17 --offload-arch=${GPU_ARCH}"
COMMON_FLAGS+=" -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2"
COMMON_FLAGS+=" -DCUDA_HAS_FP16=1 -D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1"
COMMON_FLAGS+=" -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1"
# Note: PYBIND11 defines need careful quoting for hipcc
COMMON_FLAGS+=' -DPYBIND11_COMPILER_TYPE="_gcc" -DPYBIND11_STDLIB="_libstdcpp" -DPYBIND11_BUILD_ABI="_cxxabi1011"'

# Python include path (using sysconfig for reliability)
PYTHON_INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
echo "Python include: $PYTHON_INCLUDE_PATH"

# Use arrays for proper handling of paths with spaces
INCLUDES=(
    "-I${TORCH_INCLUDE}"
    "-I${TORCH_INCLUDE}/torch/csrc/api/include"
    "-I${TORCH_INCLUDE}/TH"
    "-I${TORCH_INCLUDE}/THC"
    "-I${TORCH_INCLUDE}/THH"
    "-I${ROCM_DIR}/include"
    "-I${PYTHON_INCLUDE_PATH}"
    "-I${VENV_DIR}/include"
    "-I${SCRIPT_DIR}/third_party/glm/"
    "-I${HIP_RASTERIZER}"
)

echo "[1/5] Compiling rasterizer_impl.hip..."
"${HIPCC}" ${COMMON_FLAGS} "${INCLUDES[@]}" -c "${HIP_RASTERIZER}/rasterizer_impl.hip" -o "${BUILD_DIR}/rasterizer_impl.o"

echo "[2/5] Compiling forward.hip..."
"${HIPCC}" ${COMMON_FLAGS} "${INCLUDES[@]}" -c "${HIP_RASTERIZER}/forward.hip" -o "${BUILD_DIR}/forward.o"

echo "[3/5] Compiling backward.hip..."
"${HIPCC}" ${COMMON_FLAGS} "${INCLUDES[@]}" -c "${HIP_RASTERIZER}/backward.hip" -o "${BUILD_DIR}/backward.o"

echo "[4/5] Compiling rasterize_points.hip..."
"${HIPCC}" ${COMMON_FLAGS} "${INCLUDES[@]}" -c "${SCRIPT_DIR}/rasterize_points.hip" -o "${BUILD_DIR}/rasterize_points.o"

echo "[5/5] Compiling ext.cpp..."
g++ -fPIC -O3 -std=c++17 \
    -D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1 -DHIPBLAS_V2 \
    -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 \
    '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
    "${INCLUDES[@]}" -c "${SCRIPT_DIR}/ext.cpp" -o "${BUILD_DIR}/ext.o"

echo "[LINK] Creating shared library..."
g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions \
    -Wl,-rpath,${TORCH_LIB} -Wl,-rpath,${ROCM_DIR}/lib \
    "${BUILD_DIR}/rasterizer_impl.o" \
    "${BUILD_DIR}/forward.o" \
    "${BUILD_DIR}/backward.o" \
    "${BUILD_DIR}/rasterize_points.o" \
    "${BUILD_DIR}/ext.o" \
    -L${TORCH_LIB} -L${ROCM_DIR}/lib -L${ROCM_DIR}/hip/lib \
    -lc10 -ltorch -ltorch_cpu -ltorch_python -lamdhip64 -lc10_hip -ltorch_hip \
    -o "${BUILD_DIR}/_C.${PYTHON_SUFFIX}.so"

# Install to venv
echo "[INSTALL] Installing to venv..."
INSTALL_DIR="${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/diff_gaussian_rasterization"
mkdir -p "${INSTALL_DIR}"

# Copy Python package files
cp "${SCRIPT_DIR}/diff_gaussian_rasterization/__init__.py" "${INSTALL_DIR}/"
cp "${BUILD_DIR}/_C.${PYTHON_SUFFIX}.so" "${INSTALL_DIR}/"

echo "=== Build complete! ==="
echo "Installed to: ${INSTALL_DIR}"
