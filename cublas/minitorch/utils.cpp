#include <tuple>
#include <stdio.h>
#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"
#include <chrono>
using namespace std;


std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[])
{
    int ret1 = 0;
    int ret2 = 0;
    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-r") == 0)
        {
        }
        else if (option.compare("-s") == 0)
        {
        }
        
    }

    return {ret1, ret2};
}


