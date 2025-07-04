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
BUILD_DIR ?= .
# build for all supported CUDA architectures
CUDAARCHS != for n in $$(nvcc --list-gpu-arch); \
	do echo "$$n" | sed -e 's/.*_\([0-9]*\).*/\1/' \
	-e 's/.*/-gencode arch=compute_&,code=sm_&/'; done | tr '\n' ' '

all: $(BUILD_DIR)/gjtiff

$(BUILD_DIR)/%.o: %.cpp $(wildcard *.h *.hpp)
	$(CXX) $(CXXFLAGS) $< -c -o $@

$(BUILD_DIR)/%.o: %.c $(wildcard *.h)
	$(CC) $(CFLAGS) $< -c -o $@

$(BUILD_DIR)/%.o: %.cu %.h $(wildcard *.h *.hpp)
	$(NVCC) $(CUDAFLAGS) $(CUDAARCHS) -Xcompiler -fPIC -Xcompiler "$(CXXFLAGS)" -c $< -o $@

$(BUILD_DIR)/gjtiff: \
	$(BUILD_DIR)/downscaler.o \
	$(BUILD_DIR)/gdal_coords.o \
	$(BUILD_DIR)/kernels.o \
	$(BUILD_DIR)/libnvj2k.o \
	$(BUILD_DIR)/libnvtiff.o \
	$(BUILD_DIR)/libtiff.o \
	$(BUILD_DIR)/libtiffinfo.o \
	$(BUILD_DIR)/main.o \
	$(BUILD_DIR)/pam.o \
	$(BUILD_DIR)/rotate.o \
	$(BUILD_DIR)/utils.o
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@

clean:
	$(RM) $(BUILD_DIR)/*o $(BUILD_DIR)/gjtiff

install: $(BUILD_DIR)/gjtiff
	$(INSTALL) -m 755 $(BUILD_DIR)/gjtiff $(DESTDIR)/bin

uninstall:
	$(RM) $(DESTDIR)/bin/gjtiff

