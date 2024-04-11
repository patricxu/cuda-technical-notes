#include <tuple>
#include <stdio.h>
#include <string>
#include "utils.hpp"

using namespace std;

std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[])
{
    int iteration = 10;
    int reductionNum = 0;
    int arraySize = N;

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-i") == 0)
        {
            iteration = atoi(value.c_str());
        }
        else if(option.compare("-r") == 0)
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