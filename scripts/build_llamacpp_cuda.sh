#!/usr/bin/env bash
# Reproducible build of llama.cpp with CUDA for a GTX 1080 Ti (Pascal, sm_61),
# with NO system CUDA toolkit and NO sudo. We assemble a CUDA 12.6 toolkit from
# NVIDIA's redistributable tarballs (nvcc + cudart) plus the cu126 runtime libs
# that torch already installed under .venv (cublas/nvrtc/nvjitlink), then build.
#
# Usage:  bash scripts/build_llamacpp_cuda.sh
# Then:   source scripts/llamacpp_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NV="$ROOT/.venv/lib/python3.12/site-packages/nvidia"
TK="$ROOT/.cudatk"
CUDA="$TK/cuda"
NVCC_VER=12.6.85
CUDART_VER=12.6.77
REDIST=https://developer.download.nvidia.com/compute/cuda/redist

# 1) nvcc compiler + cccl headers (nvcc binary is NOT in the pip wheel)
uv pip install "nvidia-cuda-nvcc-cu12==12.6.*" "nvidia-cuda-cccl-cu12==12.6.*" >/dev/null
mkdir -p "$TK/dl"
[ -f "$TK/dl/nvcc.tar.xz" ]   || curl -sSL -o "$TK/dl/nvcc.tar.xz"   "$REDIST/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-${NVCC_VER}-archive.tar.xz"
[ -f "$TK/dl/cudart.tar.xz" ] || curl -sSL -o "$TK/dl/cudart.tar.xz" "$REDIST/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-${CUDART_VER}-archive.tar.xz"
tar -xJf "$TK/dl/nvcc.tar.xz"   -C "$TK"
tar -xJf "$TK/dl/cudart.tar.xz" -C "$TK"
NVCC_DIR="$TK/cuda_nvcc-linux-x86_64-${NVCC_VER}-archive"
CUDART_DIR="$TK/cuda_cudart-linux-x86_64-${CUDART_VER}-archive"

# 2) assemble a standard toolkit layout (bin / include / lib64 / nvvm)
mkdir -p "$CUDA/bin" "$CUDA/lib64" "$CUDA/include" "$CUDA/nvvm"
ln -sf "$NVCC_DIR"/bin/*  "$CUDA/bin/"
ln -sf "$NVCC_DIR"/nvvm/* "$CUDA/nvvm/"
cp -rn "$NVCC_DIR"/include/* "$CUDA/include/" 2>/dev/null || true
cp -a  "$CUDART_DIR"/lib/*.a "$CUDART_DIR"/lib/*.so* "$CUDA/lib64/"
cp -rn "$CUDART_DIR"/include/* "$CUDA/include/" 2>/dev/null || true
for c in cuda_cccl cublas cuda_nvrtc nvjitlink; do
  [ -d "$NV/$c/include" ] && cp -rn "$NV/$c/include/"* "$CUDA/include/" 2>/dev/null || true
  [ -d "$NV/$c/lib" ]     && ln -sf "$NV/$c"/lib/*.so* "$CUDA/lib64/" 2>/dev/null || true
done
( cd "$CUDA/lib64"; for b in libcublas libcublasLt libnvrtc libnvJitLink; do
    so=$(ls ${b}.so.* 2>/dev/null | head -1); [ -n "$so" ] && ln -sf "$so" "${b}.so"; done )

# 3) cmake + ninja (pip, no sudo) and clone llama.cpp
uv pip install cmake ninja >/dev/null
VBIN="$ROOT/.venv/bin"
[ -d "$ROOT/.llamacpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$ROOT/.llamacpp"

# 4) configure + build (sm_61)
cd "$ROOT/.llamacpp"
export CUDA_HOME="$CUDA" CUDACXX="$CUDA/bin/nvcc" PATH="$VBIN:$CUDA/bin:$PATH" LD_LIBRARY_PATH="$CUDA/lib64"
"$VBIN/cmake" -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" -DCMAKE_MAKE_PROGRAM="$VBIN/ninja" \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
"$VBIN/cmake" --build build --target llama-cli llama-bench llama-mtmd-cli llama-tts -j "$(nproc)"
echo "Done. Binaries in .llamacpp/build/bin/ — now: source scripts/llamacpp_env.sh"
