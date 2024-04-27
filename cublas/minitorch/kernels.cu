#include <cuda_runtime.h>
#include "utils.hpp"


__global__ void sscalKernel(const float* scalar, float* d_vector, int nVecLen){
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


void sscal(const float* scalar, float* d_vector, int nVecLen, cudaStream_t stream){
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

    sscalKernel<<<blocksPerGrid, threadPerBlock, 0, stream>>>(scalar, d_vector, nVecLen);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
}


__global__ void saxpyKernel(const float* scalar, const float* x, float* y, int nVecLen){
    // Kernel to perform saxpy operation
    // x: input vector 1
    // y: input vector 2
    // scalar: scalar to multiply with vector 1
    // nVecLen: length of the vectors
    // Each thread will multiply one element of vector 1 with the scalar
    // and add the result to the corresponding element of vector 2

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nVecLen){
        y[idx] = scalar[0] * x[idx] + y[idx];
    }
}


void saxyp(const float* scalar, const float* d_x, float* d_y, int nVecLen, cudaStream_t stream){
    // Kernel to perform saxpy operation
    // d_x: input vector 1
    // d_y: input vector 2
    // scalar: scalar to multiply with vector 1
    // nVecLen: length of the vectors
    // stream: cuda stream
    // Launch the kernel with 1D grid and block size
    // Each thread will multiply one element of vector 1 with the scalar
    // and add the result to the corresponding element of vector 2

    dim3 threadPerBlock(THREAD_PER_BLOCK);
    dim3 blocksPerGrid((nVecLen + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK);

    saxpyKernel<<<blocksPerGrid, threadPerBlock, 0, stream>>>(scalar, d_x, d_y, nVecLen);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
}