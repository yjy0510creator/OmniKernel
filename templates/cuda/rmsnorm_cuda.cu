// [OmniKernel] CUDA C++ 后端示例（RMSNorm）
// 与 Triton 路径同 spec 生成，展示 cuda 后端可选性；
// 编译: nvcc -O3 -arch=sm_80 rmsnorm_cuda.cu -shared -Xcompiler -fPIC -o rmsnorm.so
// 注：本文件为静态示例（非模板），作为 cuda 后端生成质量的参照实现。
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int WARM_BLOCK = 256;

template <typename T>
__device__ __forceinline__ float tofloat(T v);
template <>
__device__ __forceinline__ float tofloat(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float tofloat(float v) { return v; }

// 单 warp 一行；blockDim.x=32, y=8 行；Shuffle 归约平方和
__global__ void rmsnorm_warp(const __half* __restrict__ x,
                             const __half* __restrict__ w,
                             __half* __restrict__ y,
                             int rows, int hidden, float eps) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const __half* xr = x + (long)row * hidden;
    __half* yr = y + (long)row * hidden;

    float ss = 0.f;
    for (int c = threadIdx.x; c < hidden; c += 32) {
        float v = tofloat(xr[c]);
        ss += v * v;
    }
    // warp shuffle reduce
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, off);
    ss = __shfl_sync(0xffffffff, ss, 0);

    float rstd = rsqrtf(ss / hidden + eps);
    for (int c = threadIdx.x; c < hidden; c += 32) {
        float v = tofloat(xr[c]);
        yr[c] = __float2half(v * rstd * tofloat(w[c]));
    }
}

extern "C" void launch_rmsnorm(const void* x, const void* w, void* y,
                               int rows, int hidden, float eps,
                               cudaStream_t stream) {
    dim3 block(32, 8);
    int grid = (rows + 8 - 1) / 8;
    rmsnorm_warp<<<grid, block, 0, stream>>>(
        (const __half*)x, (const __half*)w, (__half*)y, rows, hidden, eps);
}
