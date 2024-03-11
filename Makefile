gjtiff: main.cpp Makefile
	g++ $< -I/usr/local/cuda/include -lcudart -lgpujpeg -lm -lnvcomp_gdeflate -lnvtiff -o $@

all: gjtiff
