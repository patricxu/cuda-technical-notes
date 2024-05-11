IDIR=./
COMPILER=nvcc
# COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17 -g -G -O0
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17 -lineinfo

.PHONY: clean build run

build: reductionadd.cu reduceAlgCascade.cu reduceAlgCascadeWarpShuffle.cu
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reductionadd.cu -o reductionadd.exe
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reduceAlgCascade.cu -o reduceAlgCascade.exe
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reduceAlgCascadeWarpShuffle.cu -o reduceAlgCascadeWarpShuffle.exe

clean:
	rm -f reductionadd.exe reduceAlgCascade.exe reduceAlgCascadeWarpShuffle.exe

run:
	./reductionadd.exe $(ARGS)

all: clean build run