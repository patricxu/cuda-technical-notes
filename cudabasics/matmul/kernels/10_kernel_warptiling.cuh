#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// #define PAD 1
#define TRANSPOSE_AS
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt {
    template <const int BM, const int BN, const int BK, const int ROWSTRIDEA,
            const int ROWSTRIDEB>
    __device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                                float *As, float *Bs, int innerRowA, int innerColA,
                                int innerRowB, int innerColB){
        for(int i = 0; i < BM; i += ROWSTRIDEA) {
            const float4 tmp = reinterpret_cast<const float4 *>(&A[(innerRowA + i) * K + innerColA * 4])[0];

#ifdef TRANSPOSE_AS
            As[(innerColA * 4 + 0) * BM + innerRowA + i] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + i] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + i] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + i] = tmp.w;
#elif PAD
            As[(innerColA * 4 + 0) * (BM + PAD) + innerRowA + i] = tmp.x;
            As[(innerColA * 4 + 1) * (BM + PAD) + innerRowA + i] = tmp.y;
            As[(innerColA * 4 + 2) * (BM + PAD) + innerRowA + i] = tmp.z;
            As[(innerColA * 4 + 3) * (BM + PAD) + innerRowA + i] = tmp.w;
#else
            reinterpret_cast<float4 *>(&As[(innerRowA + i) * BK + innerColA * 4])[0] = tmp;
#endif
        }

        for(int i = 0; i < BK; i += ROWSTRIDEB) {
            reinterpret_cast<float4 *>(Bs + (innerRowB + i) * BN + innerColB * 4)[0] = 
                reinterpret_cast<const float4 *>(B + (innerRowB + i) * N + innerColB * 4)[0];
        }
    }

    template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
    __device__ void
    processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                    const float *Bs, const uint warpRow, const uint warpCol,
                    const uint threadRowInWarp, const uint threadColInWarp) {
        for(int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) 
                for(int i = 0; i < TM; ++i) {
#ifdef TRANSPOSE_AS
                    regM[wSubRowIdx * TM + i] = As[(dotIdx * BM) + 
                        warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
#elif PAD 
                    regM[wSubRowIdx * TM + i] = As[(dotIdx * (BM + PAD)) + 
                        warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
#else
                    regM[wSubRowIdx * TM + i] = As[(warpRow * WM + wSubRowIdx * WSUBM + threadRowInWarp * TM + i) * BK + dotIdx];
#endif
                }

            for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                for(int i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] = Bs[(dotIdx * BN) + 
                        warpCol * WN + wSubColIdx * WSUBN + threadColInWarp * TN + i];
                }

            for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
                for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
                    for(int i = 0; i < TM; ++i)
                        for(int j = 0; j < TN; ++j) {
                            threadResults[(wSubRowIdx * TM + i) * WNITER * TN + 
                                wSubColIdx * TN + j] += regM[wSubRowIdx * TM + i] * regN[wSubColIdx * TN + j];
                        }
        }

    }
} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // Placement of the warp in the threadblock tile
    int warpIdx = threadIdx.x / WARPSIZE;
    int warpRow = warpIdx / (BN / WN);
    int warpCol = warpIdx % (BN / WN);

    // size of the warp subtile
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBN = WN / WNITER;
    constexpr int WSUBM = WM / WMITER;

    // Placement of the thread in the warp subtile
    int threadIdxInWarp = threadIdx.x % WARPSIZE;
    int threadColInwarp = threadIdxInWarp % (WSUBN / TN);
    int threadRowInwarp = threadIdxInWarp / (WSUBN / TN);

    // allocate space for the current blocktile in SMEM
#ifdef PAD
    __shared__ float smemA[(BM + PAD) * BK];
#else
    __shared__ float smemA[BM * BK];
#endif
    __shared__ float smemB[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    // Move C_ptr to warp's output tile
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // calculating the indices that this thread will load into SMEM
    // we'll load 128bit / 32bit = 4 elements per thread at each step
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);
    constexpr int ROWSTRIDEA = NUM_THREADS / (BK / 4);
    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);
    constexpr int ROWSTRIDEB = NUM_THREADS / (BN / 4);

    // allocate thread-local cache for results in registerfile
    float threadResults[WMITER * TM * WNITER * TN] = {0.0};
    // we cache into registers on the warptile level
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    // outer-most loop over block tiles
    for(int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // load blocktile from A and B into SMEM
        wt::loadFromGmem<BM, BN, BK, ROWSTRIDEA, ROWSTRIDEB>(
            N, K, A, B, smemA, smemB, innerRowA, innerColA, innerRowB, innerColB);

        __syncthreads();

        wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
            regM, regN, threadResults, smemA, smemB, warpRow, warpCol, threadRowInwarp, threadColInwarp);
        
        __syncthreads();
        A += BK;
        B += BK * N;
    }

    // write out the results
    for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
        for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx){
            float *C_interim = C + wSubRowIdx * WSUBM * N + wSubColIdx * WSUBN;
            for(int i = 0; i < TM; ++i)
                for(int j = 0; j < TN; j += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(&C_interim[(threadRowInwarp * TM + i) * N + 
                                                                        threadColInwarp * TN + j])[0];
                    const int base = (wSubRowIdx * TM + i) * WNITER * TN + wSubColIdx * TN + j;
                    tmp.x = alpha * threadResults[base + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[base + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[base + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[base + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(&C_interim[(threadRowInwarp * TM + i) * N + 
                                                         threadColInwarp * TN + j])[0] = tmp;
               }
        }
};