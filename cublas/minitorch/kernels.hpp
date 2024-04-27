#include <cuda_runtime.h>

void sscal(const float* scalar, float* d_vector, int nVecLen, cudaStream_t stream);

void saxyp(const float* scalar, const float* d_x, float* d_y, int nVecLen, cudaStream_t stream);
