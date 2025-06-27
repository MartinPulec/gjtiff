FROM nvidia/cuda:12.9.1-devel-ubuntu22.04
WORKDIR /build

RUN apt -y update
RUN apt -y install cmake curl gcc g++ libgdal-dev libtiff-dev make pkgconf wget
RUN apt -y install nvcomp-cuda-12 nvjpeg2k-cuda-12 nvtiff-cuda-12
RUN curl -LO https://github.com/CESNET/GPUJPEG/releases/download/continuous/\
GPUJPEG-Linux-all.tar.xz && tar xaf GPUJPEG*tar* && cp -r GPUJPEG/* /usr

COPY *c *cpp *h *hpp *cu Makefile gjtiff/
RUN cd gjtiff && make -j $(nproc) install
