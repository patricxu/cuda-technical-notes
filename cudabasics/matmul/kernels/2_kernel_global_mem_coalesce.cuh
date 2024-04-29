#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


//   dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
//   dim3 blockDim(32 * 32);
//   sgemm_global_mem_coalesce<32>
//       <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);


template <int SQRT_BLOCK_SIZE>
__global__ void sgemm_global_mem_coalesce(uint M, uint N, uint K, float alpha,
                                          const float *A, const float *B, float beta,
                                          float *C) {
                                          
    const uint xb = blockIdx.x;
    const uint yb = blockIdx.y;
    const uint threadCol = threadIdx.x % SQRT_BLOCK_SIZE;
    const uint threadRow = threadIdx.x / SQRT_BLOCK_SIZE;
    const uint cRow = xb * SQRT_BLOCK_SIZE + threadRow;
    const uint cCol = yb * SQRT_BLOCK_SIZE + threadCol;

    if(cRow >= M || cCol >= N){
        return;
    }

    float tmp = 0.0;
    for(int i = 0; i < K; ++i){
        tmp += A[cRow * K + i] * B[i * N + cCol];
    }

    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
}