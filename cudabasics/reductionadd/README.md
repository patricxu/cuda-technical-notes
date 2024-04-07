# Reduction 
The section describes add reduction in CUDA. There are 6 reducntion kernels from reduce0 to reduce5. Each version imporves on the previous. 

## Build & Run
```shell
cd reductionadd
make all
./reductionadd.exe -i 10 -r 0
```
The "-i" parameter indicates the iteration times of the function and "-r" means which reduce function to run from 0 to 5.

## The Naive Version: reduce0

Add reduction are common. In CUDA context, threads run in parallel adding the neigbor value in s distance to itself. 

![Interleaved Addressing](./pics/Screenshot%20from%202024-04-06%2018-06-28.png)

```cpp
__global__ void reduce0(int *g_idata, int *g_odata) {
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
        // printf("result=%d\n", sdata[0]);
    }
}
```
The above code is inefficient since the threads are divergent and "%" operator is very slow in the if clause.
```cpp
if (tid % (2*s) == 0)
```

## Non-divergent Version: reduce1

Replace the code
```cpp
if (tid % (2*s) == 0) {
    sdata[tid] += sdata[tid + s];
}
```
with
```cpp
int index = s * tid * 2;
if (index < blockDim.x) {
    sdata[index] += sdata[index + s];
}
```
However the non-divergent has serious bank-conflict problem in early stage of the for loop, especially when s = 1, 3. When s=1, for example, it incures a 2-way bankconfilct. ie. $$GCD(32, (1+1)mod32) = 2$$.

## Sequential Addressing: reduce2

To alleviate bank conflicts in early stages we can use sequential addressing stratage.

![Sequenial Addressing](./pics/Screenshot%20from%202024-04-06%2021-33-52.png)

```cpp
__global__ void reduce2(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
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
    }
}
```
Note that the blockDim.x is 1024.  So $s = 512$ at the beginning of the loop. $$GCD(32, (512 + 1)mod32) = 1$$, ie. no bank conflict. Every thread works in their own lane within a warp.

However, according to my experiments on GeForce 4060, the runing time of reduce2 and reduce1 are too close to tell which is better. Maybe the bank conflict problem has been optimized by morden GPUs.

We notice that half threads are idle at the beginning of the loop, which is ineffective.

## Aggressive Loading: reduce3
The reduce function can still be imporved by aggrssive loading. We can add the two values when load to shared memory. Simply replace the code
```cpp
sdata[tid] = g_idata[i];
__syncthreads();
// do reduction in shared mem
for(unsigned int s=blockDim.x / 2; s > 0; s >>= 1) {
```
with
```cpp
sdata[tid] = g_idata[i] + g_idata[blockDim.x / 2];
__syncthreads();
// do reduction in shared mem
for(unsigned int s=blockDim.x / 4; s > 0; s >>= 1) {
```

## Unrolling: reduce4
We can further imporve the it by unrolling the loop, since the loop control parameters and instructions are overhead.

```cpp
#pragma unroll
for (unsigned int s = N / 4; s > 0; s >>= 1) {
```

Note that for this pragma to work, N must be a const value so that the compiler can infer the loop number.

## Unrolling Last Warp: reduce5
Consider when $s<=32$, there are only first 32 threads working, and they are in the same warp. That means all the threads in the same warp executes in order so we need not to synchronize them and chose the active threads with "if" clause. 

```cpp
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

    if (tid < 32) {
        warpReduce(sdata, tid);
    }
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}
```

Note that the volatile key word is used to tell the compiler do not optimize so that the operation on the memory is restrictly as it ought to be, if the threads in the warp are non-divergent. However, the order of execution of sub-warps after a warp-divergence is UNDEFINED.


# Performance
Experiments were conducted on a laptop with an Intel i9 13900 CPU, 96GB RAM, and GeForce 4060 GPU with 8GB VRAM.

| Functions | Avg Time(ms) | Strategy|
|-----------|--------------|---------|
| reduce0| 0.0044|   Naive        |
| reduce1| 0.0032|  Non-divergent |
| reduce2| 0.0031|Sequential Addressing|
| reduce3| 0.0030|Aggressive Loading|
| reduce4| 0.0028|Unrolling: reduce4|
| reduce5| 0.0022|Unrolling Last Warp|

The performance gains of adopting non-divergent and unrolling-last-warp strategies are significant, whereas others show slight improvements.

# Reference
https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf
https://stackoverflow.com/questions/22939034/block-reduction-in-cuda/31730429#31730429