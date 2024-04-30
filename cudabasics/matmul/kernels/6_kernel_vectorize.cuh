#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemmVectorize(int M, int N, int K, float alpha, float *A,
                       float *B, float beta, float *C) {
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    int threadCol = threadIdx.x % (BN / TN);
    int threadRow = threadIdx.x / (BN / TN);

    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);

    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    __shared__ float smemA[BM * BK];
    __shared__ float smemB[BK * BN];

    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    float threadSum[TM][TN] = {0.0};

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    float4 tmp;

    #pragma unroll 
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        smemA[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        smemA[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        smemA[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        smemA[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        reinterpret_cast<float4*>(smemB + innerRowB * BN + innerColB * 4)[0] = reinterpret_cast<float4*>(B + innerRowB * N + innerColB * 4)[0];

        __syncthreads();
        A += BK;
        B += BK * N;

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                regM[j] = smemA[dotIdx * BM + threadRow * TM + j];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regN[j] = smemB[dotIdx * BN + threadCol * TN + j];
            }
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int k = 0; k < TN; ++k) {
                    threadSum[j][k] += regM[j] * regN[k];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) 
        #pragma unroll
        for (int j = 0; j < TN; j += 4){
            tmp = reinterpret_cast<float4*>(C + (threadRow * TM + i) * N + threadCol * TN + j)[0];
            tmp.x = alpha * threadSum[i][j + 0] + beta * tmp.x;
            tmp.y = alpha * threadSum[i][j + 1] + beta * tmp.y;
            tmp.z = alpha * threadSum[i][j + 2] + beta * tmp.z;
            tmp.w = alpha * threadSum[i][j + 3] + beta * tmp.w;

            reinterpret_cast<float4*>(C + (threadRow * TM + i) * N + threadCol * TN + j)[0] = tmp;
        }
};
