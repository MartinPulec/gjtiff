NVCC ?= nvcc
NVCC_DIR := $(shell dirname $$(command -v $(NVCC)))
COMMON = $(shell pkg-config --cflags gdal) -g -Wall -Wextra -fopenmp \
	-I$(NVCC_DIR)/../include
CFLAGS += $(COMMON)
CXXFLAGS += $(COMMON)
CUDAFLAGS ?= 
INSTALL = install
LDFLAGS += -fopenmp -L$(NVCC_DIR)/../lib64
LIBS += $(shell pkg-config --libs gdal)
LIBS += -lcudart -lgpujpeg -lm \
	-lnppc -lnppig -lnpps -lnppist \
	-lnvjpeg2k -lnvtiff -ltiff
	# -lgrok

all: gjtiff

%.o: %.cpp $(wildcard *.h *.hpp)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cu %.h $(wildcard *.h *.hpp)
	$(NVCC) $(CUDAFLAGS) -Xcompiler -fPIC -Xcompiler "$(CXXFLAGS)" -c $< -o $@

gjtiff: \
	downscaler.o \
	gdal_coords.o \
	kernels.o \
	libnvj2k.o \
	libnvtiff.o \
	libtiff.o \
	libtiffinfo.o \
	main.o \
	pam.o \
	rotate.o \
	utils.o
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@

clean:
	$(RM) *o gjtiff

install: gjtiff
	$(INSTALL) -m 755 gjtiff $(DESTDIR)/bin

uninstall:
	$(RM) $(DESTDIR)/bin/gjtiff

