#include <iostream>
#include <cuda_runtime.h>
#include "utils.hpp"
// #include "reductionadd.hpp"

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


__global__ void reduce0(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
    sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
        // printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce1(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        int index = s * tid * 2;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
        // printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
        // printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce3(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[blockDim.x/2];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=blockDim.x / 4; s > 0; s >>= 1) {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
        // printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce4(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element from global memory to shared memory
    sdata[tid] = g_idata[i] + g_idata[blockDim.x / 2];
    __syncthreads();

    // Perform reduction in shared memory using a binary tree approach
    #pragma unroll
    for (unsigned int s = N / 4; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1]; 
}

__global__ void reduce5(int *g_idata, int *g_odata) {
    extern __shared__  int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element from global memory to shared memory
    sdata[tid] = g_idata[i] + g_idata[blockDim.x / 2];
    __syncthreads();

    // Perform reduction in shared memory using a binary tree approach
    #pragma unroll
    for (unsigned int s = N / 4; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) 
    {
        warpReduce(sdata, tid);
    }
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

int main(int argc, char** argv) {
    int h_in[N];
    int h_out[N];
    auto [iteration, redctionNum] = parseCommandLineArguments(argc, argv);

    printf("begin with iteration=%d\n", iteration);

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
    for (int i = 0; i < iteration; i++){
        switch (redctionNum)
        {
        case 0:
            reduce0<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        case 1:
            reduce1<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        case 2:
            reduce2<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        case 3:
            reduce3<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        case 4:
            reduce4<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        case 5:
            reduce5<<<numBlocks, threadsPerBlock, N * sizeof(int)>>>(d_in, d_out);
            break;
        default:
            break;
        }

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    CHECK_CUDA_ERROR(cudaMemcpy((void*)h_out, (void*)d_out, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    printf("sum=%d\n", h_out[0]);
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    printf("kernel reduction%d excution time total=%f, avg=%f ms\n", redctionNum, milliseconds, milliseconds/iteration);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));

    CHECK_CUDA_ERROR(cudaFree(d_in));

    return 0;
}


