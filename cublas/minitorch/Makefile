IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda -lcublas --std c++17 -g -G -O0

.PHONY: clean build run

build: tensor.cpp
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp kernels.cu tensor.cpp -o tensor.exe

clean:
	rm -f tensor.exe

run:
	./tensor.exe $(ARGS)

all: clean build run