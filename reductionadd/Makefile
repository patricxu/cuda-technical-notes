IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: reductionadd.cu
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp reductionadd.cu -o reductionadd.exe

clean:
	rm -f reductionadd.exe output*.txt

run:
	./reductionadd.exe $(ARGS)

all: clean build run