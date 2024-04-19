#include <cuda_runtime.h>
#include "utils.hpp"


__global__ void scalarMulVectKernel(const float* scalar, float* d_vector, int nVecLen){
    // Kernel to multiply a scalar with a vector
    // d_vector: input vector
    // scalar: scalar to multiply
    // nVecLen: length of the vector
    // Each thread will multiply one element of the vector with the scalar
    // and store the result in the same location

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nVecLen){
        d_vector[idx] = scalar[0] * d_vector[idx];
    }
}


void scalarMulVect(const float* scalar, float* d_vector, int nVecLen, cudaStream_t stream){
    // Kernel to multiply a scalar with a vector
    // d_vector: input vector
    // scalar: scalar to multiply
    // nVecLen: length of the vector
    // stream: cuda stream
    // Launch the kernel with 1D grid and block size
    // Each thread will multiply one element of the vector with the scalar
    // and store the result in the same location

    dim3 threadPerBlock(THREAD_PER_BLOCK);
    dim3 blocksPerGrid((nVecLen + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

    scalarMulVectKernel<<<blocksPerGrid, threadPerBlock, 0, stream>>>(scalar, d_vector, nVecLen);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
}