FROM nvidia/cuda:12.9.1-devel-ubuntu22.04
WORKDIR /build

RUN apt -y update
RUN apt -y install curl g++ gcc git libgdal-dev libtiff-dev make pkgconf wget
RUN apt -y install nvcomp-cuda-12 nvjpeg2k-cuda-12 nvtiff-cuda-12

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v4.1.0-rc2/\
cmake-4.1.0-rc2-linux-x86_64.tar.gz && tar xaf cmake-* && cp -r cmake*/* /usr
RUN git clone https://github.com/rapidsai/cuspatial.git && \
    cd cuspatial/cpp/cuproj && cmake . && make install

RUN curl -LO https://github.com/CESNET/GPUJPEG/releases/download/continuous/\
GPUJPEG-Linux-all.tar.xz && tar xaf GPUJPEG*tar* && cp -r GPUJPEG/* /usr

COPY *c *cpp *h *hpp *cu Makefile gjtiff/
RUN cd gjtiff && CFLAGS=-DINSIDE_DHR CXXFLAGS=$CFLAGS make -j $(nproc) install
