#include <iostream>
#include <cuda_runtime.h>
#include "utils.hpp"

#define N 1024 // Size of the array

#define CHECK_CUDA_ERROR(func) \
    do { \
        cudaError_t error = (func); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void kernel(int* array, int stride, int* out) {
    __shared__ int sharedArray[N * 12];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads its corresponding element into shared memory
    sharedArray[tid] = array[tid];

    __syncthreads();

    // Simulate a read operation from shared memory
    out[tid] = sharedArray[tid] + sharedArray[tid + stride];
}

int main(int argc, char** argv) {
    int h_in[N];
    int h_out[N];
    auto [stride, iteration] = parseCommandLineArguments(argc, argv);

    printf("begin with stride=%d, iteration=%d\n", stride, iteration);

    // Initialize array data
    for (int i = 0; i < N; ++i) {
        h_in[i] = 1;
    }

    int *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, N * sizeof(int)));

    dim3 threadsPerBlock(N);
    dim3 numBlocks(1, 1);

    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    kernel<<<numBlocks, threadsPerBlock>>>(d_in, stride, d_out);
    for (int i = 0; i < iteration; i++){
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

    CHECK_CUDA_ERROR(cudaFree(d_in));

    return 0;
}
