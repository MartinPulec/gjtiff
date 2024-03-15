CXXFLAGS += -g -Wall -Wextra
NVCC ?= nvcc

all: gjtiff

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu
	$(NVCC) -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: kernels.o main.o libtiff.o
	$(CXX) $^ -lcudart -lgpujpeg -lm -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
