IDIR=./
COMPILER=nvcc
COMPILER_FLAGS=-I$(IDIR) -I/usr/local/cuda/include -lcuda --std c++17

.PHONY: clean build run

build: bankconflict.cu
	$(COMPILER) $(COMPILER_FLAGS) utils.cpp bankconflict.cu -o bankconflict.exe

clean:
	rm -f bankconflict.exe output*.txt

run:
	./bankconflict.exe $(ARGS)

all: clean build run