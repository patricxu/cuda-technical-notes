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


__global__ void kernel(int stride, int* out) {
    volatile int sharedArray[SHM_NUM];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads its corresponding element into shared memory
    for (int i = 0; i < SHM_NUM/N; i++) {
        sharedArray[tid * SHM_NUM/N + i] = 1;
    }
    __syncthreads();

    // Simulate a read operation from shared memory
    for (int i = 0; i < 10000; ++i) {
        out[tid] = sharedArray[tid] + sharedArray[(tid + stride) & (SHM_NUM-1)];
    }
}

int main(int argc, char** argv) {
    int h_out[N];
    auto [stride, iteration] = parseCommandLineArguments(argc, argv);

    printf("begin with stride=%d, iteration=%d\n", stride, iteration);

    // query shared memory bank size
    cudaSharedMemConfig sharedMemConfig;

    // set it to four, just in case
    CHECK_CUDA_ERROR(cudaDeviceGetSharedMemConfig(&sharedMemConfig));
    printf("sharedMemConfig = %d\n", sharedMemConfig);

    int *d_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, N * sizeof(int)));

    dim3 threadsPerBlock(N);
    dim3 numBlocks(1, 1);

    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iteration; i++){
        kernel<<<numBlocks, threadsPerBlock>>>(stride, d_out);
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy((void*)h_out, (void*)d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    // for(int i=0; i<N; i++)
    // {
    //     printf("h_out[%d] = %d\n", i, h_out[i]);
    // }
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    printf("kernel excution time total=%f, avg=%f ms\n", milliseconds, milliseconds/iteration);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));

    return 0;
}
