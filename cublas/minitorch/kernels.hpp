#include <cuda_runtime.h>

void scalarMulVect(const float* scalar, float* d_vector, int nVecLen, cudaStream_t stream);

