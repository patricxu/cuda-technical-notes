#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"

using namespace std;


__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1]; 
}

__global__ void reduce7(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    //DO NOT FORGET TO INITIALIZE THE SHARED MEMORY IF THE CALLED MULTIPLE TIMES
    sdata[tid] = 0;

    // Load element from global memory to shared memory
    while (i < inLen){
        sdata[tid] += g_idata[i] + g_idata[i + blockDim.x / 2];
        i += gridSize;
    }
    __syncthreads();

    // Perform reduction in shared memory using a binary tree approach
    #pragma unroll
    for (unsigned int s = N / 4; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}


void callKernel(dim3 blocks, dim3 threadsPerBlock, int *d_data, int arraySize, int *d_partialSum) {
    reduce7<<<blocks, threadsPerBlock, N * sizeof(int)>>>(d_data, arraySize, d_partialSum);

    // Check the call was successful
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }  
}


int main(int argc, char **argv) {
    auto [_, arraySize] = parseCommandLineArguments(argc, argv);

    int *h_data = new int[arraySize];
    for (int i = 0; i < arraySize; i++) {
        h_data[i] = 1;
    }

    cout << "Array size: " << arraySize << endl;
    cpuReduce(h_data, arraySize);
    
    int h_sum = 0;

    //get the sm count of the device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    dim3 blocks(min(deviceProp.multiProcessorCount, (arraySize + N - 1) / N)); 
    dim3 threadsPerBlock(N);

    int *d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, arraySize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, arraySize * sizeof(int), cudaMemcpyHostToDevice));
    
    int *d_partialSum, *d_sum;
    int dPartialSumSize = blocks.x;
    CHECK_CUDA_ERROR(cudaMalloc(&d_partialSum, dPartialSumSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_partialSum, 0, dPartialSumSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_sum, 0, sizeof(int)));

    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    callKernel(blocks, threadsPerBlock, d_data, arraySize, d_partialSum);

    if (blocks.x > 1) {
        callKernel(dim3(1, 1, 1), dim3(N), d_partialSum, dPartialSumSize, d_sum);
    } else {
        CHECK_CUDA_ERROR(cudaMemcpy(d_sum, d_partialSum, sizeof(int), cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    cout << "sum=" << h_sum << " GPU execution time total=" << milliseconds << endl;
}