# Game of Life — CUDA GPU Implementations

Design and implement a GPU version of Conway’s **Game of Life** in C/CUDA.  
This repository contains multiple CUDA kernels (global- and shared-memory approaches) alongside a simple CPU baseline for comparison, with a focus on exploring performance trade-offs between memory access patterns on NVIDIA GPUs. :contentReference[oaicite:0]{index=0}

---

## Features

- **CUDA kernels**:
  - *Simple (global memory)*: straightforward indexing and neighbor reads.
  - *Tiled (shared memory)*: block-tiled updates that reduce global reads by staging tiles + halos in shared memory.
- **CPU baseline** *(for reference & correctness checks)*.
- **Modular files** so you can compile and run each variant independently.

> File layout (by intent):
>
> - `simpleGPU.cu` — global-memory kernel  
> - `sharedGPU.cu` — shared-memory (tiled) kernel  
> - `cublas.cu` — experimental/auxiliary CUDA code (not strictly required)  
> - `cblas.c` — simple CPU reference implementation  
> 
> The repository primarily demonstrates “using both global memory and shared memory to optimize performance.” :contentReference[oaicite:1]{index=1}

---

## Requirements

- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) with `nvcc`
- A C/C++ toolchain (e.g., `gcc`/`clang`) for the CPU build

> OS: Tested easiest on Linux; should work on Windows (MSVC + NVCC) or WSL with minor adjustments.

---

## Build

Compile each variant independently with `nvcc` (and `gcc` for the CPU):

```bash
# Global-memory kernel
nvcc -O3 -arch=native -lineinfo -o gol_simple simpleGPU.cu

# Shared-memory kernel (tiled)
nvcc -O3 -arch=native -lineinfo -o gol_shared sharedGPU.cu

# (Optional) CUDA aux / cuBLAS experiment
nvcc -O3 -arch=native -o gol_cublas cublas.cu

# CPU baseline
gcc -O3 -o gol_cpu cblas.c
