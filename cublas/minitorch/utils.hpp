#include <tuple>
#include <stdio.h>
#include <string>
using namespace std;

#define THREAD_PER_BLOCK 512 // Size of the block. Must be the power of 2
#define MAX_BLOCK_SIZE 1024

#define CHECK_CUDA_ERROR(func) \
    do { \
        cudaError_t error = (func); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[]);
void cpuReduce(int *in, int inLen);