#include <iostream>
#include <cuda_runtime.h>
#include "utils.hpp"
// #include "reductionadd.hpp"


#define CHECK_CUDA_ERROR(func) \
    do { \
        cudaError_t error = (func); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void reduce0(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    if (tid == 0){
        g_odata[blockIdx.x] = sdata[0];
        printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce1(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


__global__ void reduce2(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
        printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce3(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x / 2];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s = blockDim.x / 4; s > 0; s >>= 1) {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) 
    {
        g_odata[blockIdx.x] = sdata[0];
        printf("result=%d\n", sdata[0]);
    }
}


__global__ void reduce4(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element from global memory to shared memory
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x / 2];
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

__global__ void reduce5(int *g_idata, int inLen, int *g_odata) {
    extern __shared__  int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element from global memory to shared memory
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x / 2];
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
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


template <unsigned int blockSize>
__device__ void warpReduce1(volatile int* sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=2) sdata[tid] += sdata[tid + 1]; 
}


template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int inLen, int *g_odata) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x / 2];
    __syncthreads();

    if (blockSize > 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
    }

    if (blockSize > 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
    }

    if (blockSize > 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }
    }

    if (tid < 32) {
        warpReduce1<blockSize>(sdata, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}


void callKernel(int reductioinNum, int *d_in, int inLen, int *d_out, dim3 blockDim, dim3 threadDim) {
    printf("call kernel reductioinNum=%d, inLen=%d, blockDim.x=%d, threadDim.x=%d\n", reductioinNum, inLen, blockDim.x, threadDim.x);
    switch (reductioinNum)
        {
        case 0:
            reduce0<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 1:
            reduce1<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 2:
            reduce2<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 3:
            reduce3<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 4:
            reduce4<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 5:
            reduce5<<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
            break;
        case 6:
            reduce6<N><<<blockDim, threadDim, N * sizeof(int)>>>(d_in, inLen, d_out);
        default:
            break;
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int *h_in;
    int *h_out;
    int numBlocks = 0;
    int *d_partialSum;
    auto [reductioinNum, arraySize] = parseCommandLineArguments(argc, argv);

    printf("begin with reduction%d, arraySize=%d\n", reductioinNum, arraySize);

    int upBndArraySizePwOfTwo = 1;
    do{
        upBndArraySizePwOfTwo <<= 1;
    } while (upBndArraySizePwOfTwo < arraySize);

    h_in = (int*)malloc(upBndArraySizePwOfTwo * sizeof(int));

    numBlocks = (upBndArraySizePwOfTwo + N - 1) / N;

    // Initialize array data
    for (int i = 0; i < upBndArraySizePwOfTwo; ++i) {
        if (i < arraySize) h_in[i] = 1;
        else h_in[i] = 0;
    }

    int sum = 0;
    for (int i = 0; i < upBndArraySizePwOfTwo; i++) {
        sum += h_in[i];
    }
    printf("sum=%d\n", sum);

    int *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, upBndArraySizePwOfTwo * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, upBndArraySizePwOfTwo * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_partialSum, numBlocks * sizeof(int)));

    printf("The size of extended input array is %d\n", upBndArraySizePwOfTwo);
    printf("The size of partial sum array is %d\n", numBlocks);

    dim3 threadDim(N);
    dim3 blockDim(numBlocks, 1);

    cudaEvent_t start, end;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&end));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    int * d_tmp_in = d_in;
    int * d_tmp_out = d_partialSum;
    for (int j = upBndArraySizePwOfTwo; j > 1; j /= threadDim.x){
        callKernel(reductioinNum, d_tmp_in, j, d_tmp_out, blockDim, threadDim);
        d_out = d_tmp_out;
        d_tmp_out = d_tmp_in;
        d_tmp_in = d_out;
        blockDim.x = blockDim.x / threadDim.x + 1;
    }

    CHECK_CUDA_ERROR(cudaMemcpy((void*)h_out, (void*)d_out, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(end));
    CHECK_CUDA_ERROR(cudaEventSynchronize(end));

    printf("sum=%d\n", h_out[0]);
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    printf("kernel reduction%d excution time total=%f\n", reductioinNum, milliseconds);

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(end));
    CHECK_CUDA_ERROR(cudaFree(d_partialSum));
    CHECK_CUDA_ERROR(cudaFree(d_in));

    return 0;
}


