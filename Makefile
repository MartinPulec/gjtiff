NVCC ?= nvcc
NVCC_DIR := $(shell dirname $$(command -v $(NVCC)))
CFLAGS += -g -Wall -Wextra -fopenmp -I$(NVCC_DIR)/../include
CXXFLAGS += $(CFLAGS)
CUDAFLAGS ?= 
LDFLAGS += -fopenmp -L$(NVCC_DIR)/../lib64

all: gjtiff

%.o: %.cpp $(wildcard *.h *.hpp)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cu %.h $(wildcard *.h *.hpp)
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: downscaler.o kernels.o libnvj2k.o libnvtiff.o libtiff.o libtiffinfo.o main.o utils.o
	$(CXX) $(LDFLAGS) $^ -lcudart -lgpujpeg -lm -lnppc -lnppig -lnpps -lnppist -lnvjpeg2k -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
