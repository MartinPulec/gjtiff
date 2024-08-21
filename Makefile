NVCC ?= nvcc
NVCC_DIR := $(shell dirname $$(command -v $(NVCC)))
CXXFLAGS += -g -Wall -Wextra -Wno-missing-field-initializers -I$(NVCC_DIR)/../include
CUDAFLAGS ?= 
LDFLAGS += -L$(NVCC_DIR)/../lib64

all: gjtiff

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: kernels.o libnvtiff.o libtiff.o main.o
	$(CXX) $(LDFLAGS) $^ -lcudart -lgpujpeg -lm -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
