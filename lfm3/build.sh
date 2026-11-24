#!/bin/bash
# build.sh : compile build/lfm3-engine pour Radeon (HIP).   ROCM=<racine ROCm> [GPU_TARGET=gfx1201] ./build.sh
set -e
cd "$(dirname "$0")"
R=${ROCM:-/opt/rocm}; T=${GPU_TARGET:-gfx1201}
mkdir -p build
# voir inkling2/build.sh : les en-têtes hip/ d'un autre ROCm dans /usr/include passeraient devant
rm -rf build/rocm-include; mkdir -p build/rocm-include; cp -a "$R/include/hip" build/rocm-include/
"$R/bin/hipcc" --rocm-path="$R" -Ibuild/rocm-include -x hip -O3 -std=c++17 --offload-arch="$T" \
  src/lfm3_engine.cu -o build/lfm3-engine -L"$R/lib" -lhipblas -Wl,-rpath,"$R/lib" "$@"
echo "build/lfm3-engine ($T)"
