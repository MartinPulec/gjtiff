CXXFLAGS += -g -Wall -Wextra

all: gjtiff

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c -I/usr/local/cuda/include -o $@

%.o: %.cu
	nvcc -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: kernels.o main.o libtiff.o
	$(CXX) $^ -lcudart -lgpujpeg -lm -lnvcomp_gdeflate -lnvtiff -ltiff -o $@

clean:
	$(RM) *o gjtiff
