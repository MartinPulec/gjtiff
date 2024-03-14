# GJTIFF

Tool to convert _TIFF_ files to _JPEGs_ on the **GPU**
utilizing [GPUJPEG](https://github.com/CESNET/GPUJPEG) and
[nvTIFF](https://developer.nvidia.com/nvtiff-downloads).
[nvCOMP](https://developer.nvidia.com/nvcomp) is also required for
DEFLATE decompress.

If nvTIFF isn't capable to decompress the picture _libtiff_ is used
instead.

# Requirements

Build:

- CUDA + C++ compiler + make
- [GPUJPEG](https://github.com/CESNET/GPUJPEG)
- [nvTIFF](https://developer.nvidia.com/nvtiff-downloads)
- [libTIFF](https://libtiff.gitlab.io/libtiff)

Runtime:

- CUDA device
- [nvCOMP](https://developer.nvidia.com/nvcomp-download) for DEFLATE

# Performance

Using _AMD Ryzen 9 7900X_ and _GeForce RTX 4080_:

- 8K x 8K - 30 ms
- 8K x 8K - 200 ms using CPU libtiff decoder
- 16K x 16K - 96 ms (450 ms with initialization/reconfiguration)
