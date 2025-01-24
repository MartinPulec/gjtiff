NVCC ?= nvcc
NVCC_DIR := $(shell dirname $$(command -v $(NVCC)))
COMMON = $(shell pkg-config --cflags libgrokj2k) -g -Wall -Wextra -fopenmp \
	-I$(NVCC_DIR)/../include
CFLAGS += $(COMMON)
CXXFLAGS += $(COMMON)
CUDAFLAGS ?= 
LDFLAGS += -fopenmp -L$(NVCC_DIR)/../lib64

all: gjtiff

%.o: %.cpp $(wildcard *.h *.hpp)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cu %.h $(wildcard *.h *.hpp)
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: downscaler.o kernels.o libnvj2k.o libnvtiff.o libtiff.o libtiffinfo.o main.o rotate.o utils.o
	$(CXX) $(LDFLAGS) $^ -lcudart -lgpujpeg -lgrokj2k -lm -lnppc -lnppig -lnpps -lnppist -lnvjpeg2k -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
