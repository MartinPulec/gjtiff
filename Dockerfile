FROM debian:12
WORKDIR /build

RUN apt-get -y update
RUN apt-get -y install curl g++ gcc git libgdal-dev libtiff-dev libwebp-dev \
                   make pkgconf wget xz-utils

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/\
debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get -y update
RUN apt-get -y install cuda-compiler-12-9 libnpp-dev-12-9
RUN apt-get -y install nvcomp nvjpeg2k-cuda-12 nvtiff-cuda-12
ENV CUDACXX=/usr/local/cuda/bin/nvcc

RUN curl -LOSs https://github.com/Kitware/CMake/releases/download/v4.1.0-rc2/\
cmake-4.1.0-rc2-linux-x86_64.tar.gz && tar xaf cmake-* && cp -r cmake*/* /usr
RUN git clone https://github.com/rapidsai/cuspatial.git && \
    cd cuspatial/cpp/cuproj && cmake -Wno-dev . && make install

RUN curl -LOSs https://github.com/CESNET/GPUJPEG/releases/download/continuous/\
GPUJPEG-Linux-all.tar.xz && tar xaf GPUJPEG*tar* && cp -r GPUJPEG/* /usr

COPY .git *c *cpp *h *hpp *cu *cuh Makefile gjtiff/
RUN cd gjtiff && CFLAGS=-DINSIDE_DHR CXXFLAGS=$CFLAGS make -j $(nproc) install
