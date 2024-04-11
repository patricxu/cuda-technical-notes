#include <tuple>
#include <stdio.h>
#include <string>
using namespace std;

#define N 512 // Size of the block. Must be the power of 2
#define MAX_BLOCK_SIZE 1024

std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[]);