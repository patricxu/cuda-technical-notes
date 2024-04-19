IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17 -g -G -O0

.PHONY: clean build run

build: reductionadd.cu reduceAlgCascade.cu
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reductionadd.cu -o reductionadd.exe
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reduceAlgCascade.cu -o reduceAlgCascade.exe

clean:
	rm -f reductionadd.exe reduceAlgCascade.exe

run:
	./reductionadd.exe $(ARGS)

all: clean build run