#include <tuple>
#include <stdio.h>
#include <string>
using namespace std;

std::tuple<int, int> parseCommandLineArguments(int argc, char *argv[])
{
    int stride = 0;
    int iteration = 10;

    for(int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if(option.compare("-s") == 0) 
        {
            stride = atoi(value.c_str());
        }
        else if(option.compare("-i") == 0)
        {
            iteration = atoi(value.c_str());
        }

    }

    return {stride, iteration};
}