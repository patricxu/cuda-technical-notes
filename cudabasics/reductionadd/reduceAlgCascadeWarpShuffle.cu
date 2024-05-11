#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"

#define WARP_SIZE 32
#define MASK 0xFFFFFFFF

using namespace std;


__global__ void reduce8(int *g_idata, int inLen, int *g_odata) {
    __shared__ int sdata[WARP_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int gTid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = gTid;
    unsigned int gridSize = blockDim.x * gridDim.x;
    unsigned int lane = tid % WARP_SIZE;
    unsigned int warpId = tid / WARP_SIZE;
    int val = 0;

    while (i < inLen){
        val += g_idata[i];
        i += gridSize;
    }

    //warp level reduce
    #pragma unroll
    for (unsigned int s = WARP_SIZE / 2; s > 0; s >>= 1) {
        val += __shfl_down_sync(MASK, val, s);
    }

    if (lane == 0){
        sdata[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0){
        val = (tid < blockDim.x / WARP_SIZE) ? sdata[lane] : 0;

        //block level reduce
        for (unsigned int s = WARP_SIZE / 2; s > 0; s >>= 1) {
            val += __shfl_down_sync(MASK, val, s);
        }    
            //grid level reduce
        if (tid == 0){
            atomicAdd(g_odata, val);
        }
    }
}


void callKernel(dim3 blocks, dim3 threadsPerBlock, int *d_data, int arraySize, int *d_partialSum) {
    reduce8<<<blocks, threadsPerBlock, N * sizeof(int)>>>(d_data, arraySize, d_partialSum);

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

    std::cout << "Array size: " << arraySize << endl;
    cpuReduce(h_data, arraySize);
    
    int h_sum = 0;

    //get the sm count of the device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, dev));
    int numBlocks;
    CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, reduce8, N, WARP_SIZE * sizeof(int)));
    std::cout << "occupancy max active blocks per sm is " << numBlocks << endl;
    // dim3 blocks(min(deviceProp.multiProcessorCount, (arraySize + N - 1) / N)); 
    dim3 blocks(numBlocks * deviceProp.multiProcessorCount);
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
    std::cout << "blocks.x=" << blocks.x << " threadsPerBlock=" << threadsPerBlock.x << endl;
    callKernel(blocks, threadsPerBlock, d_data, arraySize, d_partialSum);
    CHECK_CUDA_ERROR(cudaMemcpy(d_sum, d_partialSum, sizeof(int), cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "sum=" << h_sum << " GPU execution time total=" << milliseconds << endl;
}