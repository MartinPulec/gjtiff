CXXFLAGS += -g -Wall -Wextra -Wno-missing-field-initializers
NVCC ?= nvcc

all: gjtiff

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu
	$(NVCC) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: kernels.o libnvtiff.o libtiff.o main.o
	$(CXX) $^ -lcudart -lgpujpeg -lm -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
