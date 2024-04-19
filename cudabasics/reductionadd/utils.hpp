#include <tuple>
#include <stdio.h>
#include <string>
using namespace std;

#define N 512 // Size of the block. Must be the power of 2
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


std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[]);
void cpuReduce(int *in, int inLen);