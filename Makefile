all: gjtiff

main.o: main.cpp
	g++ $< -c -I/usr/local/cuda/include -o $@

kernels.o: kernels.cu
	nvcc -c $< -o $@

gjtiff: kernels.o main.o
	g++ $^ -lcudart -lgpujpeg -lm -lnvcomp_gdeflate -lnvtiff -o $@

