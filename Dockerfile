FROM debian:12
WORKDIR /build

RUN sed -i 's/^Components:.*/Components: main contrib non-free non-free-firmware/' \
  /etc/apt/sources.list.d/debian.sources
RUN apt -y update
RUN apt -y install build-essential cmake git libgdal-dev libtiff-dev make pkgconf wget
RUN apt -y --no-install-recommends install nvidia-cuda-toolkit

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/\
debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt -y update
RUN apt -y install nvcomp nvjpeg2k-cuda-11 nvtiff-cuda-11

RUN git clone https://github.com/CESNET/GPUJPEG.git && \
  cmake -BGPUJPEG/build -D CMAKE_INSTALL_PREFIX=/usr GPUJPEG && \
  cmake --build GPUJPEG/build -j $(nproc) && \
  cmake --install GPUJPEG/build

RUN git clone https://github.com/MartinPulec/gjtiff.git && \
  cd gjtiff && make -j $(nproc) install
