#!/bin/bash
# build-cuda.sh : compile build/lfm3-engine pour NVIDIA.   [CUDA_ARCH=86] [CXX=g++-14] ./build-cuda.sh
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
A=${CUDA_ARCH:-86}
mkdir -p build
"$NVCC" -O3 -std=c++17 -arch=sm_$A ${CXX:+-ccbin "$CXX"} -DATTN_KT=${ATTN_KT:-64} src/lfm3_engine.cu -o build/lfm3-engine -lcublas "$@"
echo "build/lfm3-engine (sm_$A)"
