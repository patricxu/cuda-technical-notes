#include <tuple>
#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"
#include <chrono>
using namespace std;


void cpuReduce(int *in, int inLen) {
    auto start = std::chrono::high_resolution_clock::now();

    int sum = 0;
    for (int i = 0; i < inLen; i++)
        sum += in[i];
    
    // Timing end
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate execution time in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double milliseconds = duration.count();

    // Print the execution time
    std::cout << "sum=" << sum << " CPU Execution time: " << milliseconds << " milliseconds" << std::endl;
}


std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[])
{
    int reductionNum = 0;
    int arraySize = N;

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-r") == 0)
        {
            reductionNum = atoi(value.c_str());
        }
        else if (option.compare("-s") == 0)
        {
            arraySize = atoi(value.c_str());
        }
        
    }

    return {reductionNum, arraySize};
}


