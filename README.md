# GJTIFF

Tool to convert _TIFF_ files to _JPEGs_ on the CUDA-capable **GPU**
utilizing [GPUJPEG](https://github.com/CESNET/GPUJPEG) and
[nvTIFF](https://developer.nvidia.com/nvtiff-downloads).
[nvCOMP](https://developer.nvidia.com/nvcomp) is also required for
DEFLATE decompress.

If nvTIFF isn't capable to decompress the picture _libtiff_ is used
instead.

# Requirements

## Software

Build:

- CUDA toolkit + C++ compiler + make
- [GPUJPEG](https://github.com/CESNET/GPUJPEG)
- [nvTIFF](https://developer.nvidia.com/nvtiff-downloads)
- [libTIFF](https://libtiff.gitlab.io/libtiff)

Runtime:

- NVIDIA proprietary drivers
- [nvCOMP](https://developer.nvidia.com/nvcomp-download) library for DEFLATE

## Hardware

- CUDA-capable device - GPUJPEG requires approximately 60 B of memory for a
pixel, so 16K x 16K requires approximately 16 GB RAM.

# Performance

Using _AMD Ryzen 9 7900X_ and _NVIDIA GeForce RTX 4080_:

- 8K x 8K - 30 ms
- 8K x 8K - 200 ms using CPU libtiff decoder
- 16K x 16K - 96 ms (450 ms with initialization/reconfiguration)

As with GPUJPEG, it is advisable to process multiple images in a batch
than calling it for a single image to amortize the initialization cost.

# TODO

- configurable JPEG parameters (especially quality)
- improved libtiff decoding performance (if needed)
- option to select CUDA device
- various performance optimizations - eg. when image sizes change
often, which is slow due to GPUJPEG reconfiguration

# Sample images' notes

- Copernicus/download/Landsat-8/OLI_TIRS/L1TP semm to be in visible range
- Sentinel-1 need to have the levels adjusted
- Sentinel-2 in J2K (the support needs to be added)
