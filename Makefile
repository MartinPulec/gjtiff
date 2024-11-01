NVCC ?= nvcc
NVCC_DIR := $(shell dirname $$(command -v $(NVCC)))
CXXFLAGS += -g -Wall -Wextra -fopenmp -I$(NVCC_DIR)/../include
CUDAFLAGS ?= 
LDFLAGS += -fopenmp -L$(NVCC_DIR)/../lib64

all: gjtiff

%.o: %.cpp $(wildcard *.h *.hpp)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu %.hpp
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: kernels.o libnvj2k.o libnvtiff.o libtiff.o libtiffinfo.o main.o
	$(CXX) $(LDFLAGS) $^ -lcudart -lgpujpeg -lm -lnppc -lnpps -lnppist -lnvjpeg2k -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
