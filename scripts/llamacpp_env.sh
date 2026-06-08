#!/usr/bin/env bash
# Source this to put the locally-built llama.cpp (CUDA, sm_61) on PATH with the
# right CUDA runtime libs, on a machine with NO system CUDA toolkit and no sudo.
#
#   source scripts/llamacpp_env.sh
#   llama-bench -m gguf/text/LFM2.5-1.2B-JP-202606-Q8_0.gguf -ngl 99
#
# The CUDA toolkit under .cudatk/ was assembled from NVIDIA redistributable
# tarballs (nvcc, cudart) + the cu126 libs torch already ships in .venv. See
# scripts/build_llamacpp_cuda.sh and docs/acceleration.md.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$ROOT/.cudatk/cuda"
export PATH="$ROOT/.llamacpp/build/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
