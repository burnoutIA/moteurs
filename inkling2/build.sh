#!/bin/bash
# build.sh : compile strata-ink2 (portage HIP) dans build/.
#   ROCM=<racine ROCm, 7.x ou plus> [GPU_TARGET=gfx1201] ./build.sh [options hipcc en plus]
# Testé avec ROCm 10.1 (TheRock) sur gfx1201 (Radeon AI PRO R9700).
set -e
cd "$(dirname "$0")"
R=${ROCM:-/opt/rocm}
T=${GPU_TARGET:-gfx1201}
mkdir -p build
# Si un autre ROCm est installé dans /usr/include, hipcc met ses en-têtes hip/ devant ceux de $R, et rocWMMA prend
# le mauvais hip_fp8.h (erreurs __ocml_*). Un -I sur $R/include est écarté comme doublon d'un dossier système,
# même par un lien symbolique : on copie donc hip/ ailleurs, pour qu'il passe devant.
rm -rf build/rocm-include; mkdir -p build/rocm-include; cp -a "$R/include/hip" build/rocm-include/
"$R/bin/hipcc" --rocm-path="$R" -Ibuild/rocm-include -Isrc -x hip -O3 -std=c++17 --offload-arch="$T" \
  -fgpu-default-stream=per-thread src/ink2_engine.hip -o build/strata-ink2 -L"$R/lib" -lhipblas -Wl,-rpath,"$R/lib" "$@"
echo "build/strata-ink2 ($T)"
