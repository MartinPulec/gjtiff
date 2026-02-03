# GJTIFF

Tool to convert _TIFF_ files to _JPEGs_ on the CUDA-capable **GPU**
utilizing [GPUJPEG](https://github.com/CESNET/GPUJPEG) and
[nvTIFF](https://developer.nvidia.com/nvtiff-downloads).
[nvCOMP](https://developer.nvidia.com/nvcomp) is also required for
DEFLATE decompress.

If nvTIFF isn't capable to decompress the picture _libtiff_ is used
instead.

# NEWS

- 12th Nov '25 - add parameter `-W` for encoding tiles as WebP and whole as JPG
- 11th Nov '25 - add parameter `-F` to change fill-color from black to eg. white
- Oct '25 - support for more correct S1 repositioning to WebP (using
GeoTIFF tie pointes)
- 1st Aug '25 - added -w to generate WebP (including alpha); added -N
to suppress generating whole image (in tiled mode) - performance reasons
- 29th Jun '25 - jpeg2000 decode performance improved through
parallel tile decoding

# Requirements

## Software

Build:

- CUDA toolkit (version at least 12), NPP (distributed in CUDA)
- C++ compiler, make, pkg-config/pkgconf
- [GPUJPEG](https://github.com/CESNET/GPUJPEG)
- [nvJPEG2000](https://developer.nvidia.com/nvjpeg2000-downloads)
- [nvTIFF](https://developer.nvidia.com/nvtiff-downloads)
- [libTIFF](https://libtiff.gitlab.io/libtiff)
- [cuspatial](https://github.com/rapidsai/cuspatial)

Runtime:

- NVIDIA proprietary drivers (supporting CUDA >= 12)
- the libraries of the build deps unless linked with static versions
- [nvCOMP](https://developer.nvidia.com/nvcomp-download) library for
DEFLATE suppot in nvTIFF

## Hardware

- CUDA >= 12 capable device (Maxwell microarchitecture onwards)
- GPUJPEG requires approximately 60 B of memory for RGB pixel, so 16K x
16K requires approximately 16 GB VRAM

# Build and running

Build using Docker:
```
docker build -f Dockerfile -t gjtiff_image .
```

Run:
```
docker run --gpus all --rm -v ~/data:/data gjtiff_image gjtiff -o /data /data/image.jp2
```

**Note:** _nvidia-container-toolkit_ should be installed (Debian) to
pass-through the NVIDIA GPU.

# Usage

In case of the combined features (RGB mapped, normalized differential),
the input tiles/pictures must be single channel.

## Combined features - RGB mapped

Input files format (in plae of input file name): `fname1,fname2,fname3`
output is `oname1-COMMA-oname2-COMMA-oname3.ext` where onameX=$(basename
${fnameX%.*}) (just filename without path and extension); _ext_ is jpeg
or webp. There must be **3** input files.

## Combined features - normalized differential

(S2 only)

Currently supported features are **NDVI**, **NDWI** and **NDMI**
and generic **ND_UNSPEC** (rendered as a grayscale).  All of them
currently require 2 file names, the syntax is `feature@fname1,fname2`
where  the feature name is one of the above. Output file name is
`feature-oname1-COMMA-oname2.ext` where the oname is specified in
previous section (**note** between _feature_ and _oname1_ is '-'
not at-sign).

# Supported input formats

- TIFF (decodable either by nvtiff or libtiff as a fallback)
- J2K - if decodable by nvjpeg2000 (seems to be the case for all
Copernicus images), no fallback by now

# Performance

Using _AMD Ryzen 9 7900X_ and _NVIDIA GeForce RTX 4080_:

- 8K x 8K - 30 ms
- 8K x 8K - 200 ms using CPU libtiff decoder
- 16K x 16K - 96 ms (450 ms with initialization/reconfiguration)

As with GPUJPEG, it is advisable to process multiple images in a batch
than calling it for a single image to amortize the initialization cost.

# TODO

- improved nvjpeg2000 decoding performance if needed
<https://developer.nvidia.com/blog/accelerating-jpeg-2000-decoding-for-digital-pathology-and-satellite-images-using-the-nvjpeg2000-library/>
- option to select CUDA device
- various performance optimizations - eg. when image sizes change
often, which is slow due to GPUJPEG reconfiguration
- cudaMalloc/Free Async version should be used when used within
streams (as deployed now, it doesn't cause probbems right now)

# Postprocess - output normalization/equalization

The images are converted to visual representation by equalizing the input
as present in _SNAP_ and applying **gamma=2** (not present in SNAP). The
equalization is performed to scale input [0, µ + 2σ] to output [0,
255]. If maximal sample in the set is less than µ  + 2σ, this is used
instead (not likely but happens sometimes when the input values are very
small). The value above are clamped to 255.

For **3-channel J2K images** the normalization (neither Gamma-compression)
doesn't take place because those seem to represent visual-range data
(including the Gamma). Note that SNAP handles the image bands separately
(as a grayscale) and it also undergoes the equalization (by default).


# Notes and issues

- NDVI rendering doesn't use reference scale (see the '@^{/fix NDVI}' commit)

# Sample images' notes

- Copernicus/download/Landsat-8/OLI_TIRS/L1TP semm to be in visible range
- Sentinel-1 need to have the levels adjusted
- Sentinel-2 in J2K (the support needs to be added)
