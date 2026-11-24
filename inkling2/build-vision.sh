#!/bin/bash
# build-vision.sh : compile build/inkling2-vision, l'encodeur d'image d'Inkling 2 Mini (mmproj de llama.cpp, seul).
#   ROCM=<racine ROCm> [GPU_TARGET=gfx1201] ./build-vision.sh          encodeur sur la Radeon (HIP)
#   VISION_BACKEND=cpu ./build-vision.sh                               encodeur sur le processeur (NVIDIA : la carte
#                                                                      garde sa mémoire pour le moteur)
# llama.cpp est figé au commit LLAMA_COMMIT (après #32217, « Inkling 2 support (text + vision) »), avec
# vision/mtmd-sans-vocab.patch : mtmd encode alors une image sans modèle de texte. Seule la cible mtmd est compilée.
# Sans réseau (une bulle) : extraire avant l'archive de ce commit dans build/llama.cpp, le script la prend telle quelle.
# L'encodeur lui-même : poids/mmproj-BF16.gguf, à prendre dans unsloth/Inkling-2-Mini-GGUF (0,64 Go).
set -e
cd "$(dirname "$0")"
LLAMA_COMMIT=c74c71a6f7a0b06df30f979520e5d9a86a4d60ad
B=${VISION_BACKEND:-hip}
L=build/llama.cpp
mkdir -p build
if [ -d "$L/.git" ]; then git -C "$L" checkout -q -f "$LLAMA_COMMIT"
elif [ ! -f "$L/CMakeLists.txt" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$L"
  git -C "$L" checkout -q -f "$LLAMA_COMMIT"
fi
P="$PWD/vision/mtmd-sans-vocab.patch"
if ! (cd "$L" && patch -p1 -R -s -f --dry-run < "$P" >/dev/null 2>&1); then (cd "$L" && patch -p1 -s < "$P"); fi
ARGS=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_SERVER=OFF
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF)
if [ "$B" = hip ]; then
  R=${ROCM:-/opt/rocm}
  T=${GPU_TARGET:-gfx1201}
  export PATH="$R/bin:$R/llvm/bin:$PATH" ROCM_PATH="$R" HIP_PATH="$R" HIP_PLATFORM=amd
  # -isystem : le clang de ROCm range ses en-têtes après /usr/include, où un autre ROCm peut avoir les siens (voir build.sh)
  ARGS+=(-DGGML_HIP=ON -DGPU_TARGETS="$T" -DCMAKE_HIP_COMPILER="$R/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$R"
         -DCMAKE_PREFIX_PATH="$R" -Dhip_DIR="$R/lib/cmake/hip" "-DCMAKE_HIP_FLAGS=--rocm-path=$R -isystem $R/include"
         "-DCMAKE_CXX_FLAGS=-isystem $R/include")
elif [ "$B" != cpu ]; then echo "VISION_BACKEND : hip ou cpu" >&2; exit 2; fi
cmake -S "$L" -B "$L/build" "${ARGS[@]}" >/dev/null
cmake --build "$L/build" -j "$(nproc)" --target mtmd
g++ -O2 -std=c++17 -I"$L/tools/mtmd" -I"$L/include" -I"$L/ggml/include" vision/inkling2-vision.cpp -o build/inkling2-vision \
  -L"$L/build/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin'
echo "build/inkling2-vision (llama.cpp ${LLAMA_COMMIT:0:10}, $B)"
