all: gjtiff

%.o: %.cpp
	g++ $< -c -I/usr/local/cuda/include -o $@

kernels.o: kernels.cu
	nvcc -c $< -o $@

gjtiff: kernels.o main.o libtiff.o
	g++ $^ -lcudart -lgpujpeg -lm -lnvcomp_gdeflate -lnvtiff -ltiff -o $@

