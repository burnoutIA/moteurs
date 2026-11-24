#!/bin/bash
# build-cuda.sh : compile strata-ink2 pour NVIDIA dans build/, depuis les sources CUDA d'origine (amont/, inchangées),
# auxquelles build/ applique vision/ink2-images-cuda.patch (les images, comme dans le portage HIP : IMG, IMGRESET).
#   [CUDA_ARCH=86] [CXX=g++-14] ./build-cuda.sh [options nvcc en plus]
# Testé avec CUDA 13.4 sur une RTX 3090 (sm_86). CUDA_ARCH : 86 pour une RTX 30, 89 pour une RTX 40, 120 pour une RTX 50.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
A=${CUDA_ARCH:-86}
mkdir -p build
cp amont/engine/src/inkling2/ink2_engine.cu build/ink2_engine.cu
(cd build && patch -p1 -s < ../vision/ink2-images-cuda.patch)
# nvcc n'accepte qu'une plage de g++ : CXX=g++-N quand celui du système est trop récent.
"$NVCC" -O3 -std=c++17 -arch=sm_$A --default-stream per-thread ${CXX:+-ccbin "$CXX"} \
  -Iamont/engine/src/inkling2 build/ink2_engine.cu -o build/strata-ink2 -lcublas "$@"
echo "build/strata-ink2 (sm_$A)"
