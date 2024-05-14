#include <iostream>
#include <cuda_runtime.h>
#include "utils.hpp"

#define N 32 // Threads number. The max number of threads in a block is 1024
#define SHM_NUM N * N 

#define CHECK_CUDA_ERROR(func) \
    do { \
        cudaError_t error = (func); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void kernel(int stride, unsigned long long *d_out) {
    __shared__ int sharedArray[SHM_NUM];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long startClock = clock64();
    sharedArray[tid * stride]++;
    unsigned long long endClock = clock64();

    d_out[tid] = endClock - startClock;
}

int main(int argc, char** argv) {
    unsigned long long h_out[N];
    auto [stride, _] = parseCommandLineArguments(argc, argv);

    printf("begin with stride=%d\n", stride);

    unsigned long long *d_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, N * sizeof(unsigned long long)));

    dim3 threadsPerBlock(N);
    dim3 numBlocks(1, 1);

    kernel<<<numBlocks, threadsPerBlock>>>(stride, d_out);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaMemcpy((void*)h_out, (void*)d_out, N * sizeof(unsigned long long), cudaMemcpyDeviceToHost));


    for(int i=0; i<N; i++)
    {
        printf("h_out[%d] = %llu\n", i, h_out[i]);
    }
    

    return 0;
}
