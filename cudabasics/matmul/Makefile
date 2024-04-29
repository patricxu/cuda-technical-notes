IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda -lcublas --std c++17 -O3

.PHONY: clean build run

build: sgemm.cu
	$(COMPILER) $(COMPILER_FLAGS) runner.cu sgemm.cu -o sgemm.exe

clean:
	rm -f sgemm.exe 

run:
	./sgemm.exe $(ARGS)

all: clean build run