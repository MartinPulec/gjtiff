# GJTIFF

Tool to convert Copernicus S1 (TIFF) and S2 (jp2) files to _JPEGs_ on the CUDA-capable
**GPU** utilizing [GPUJPEG](https://github.com/CESNET/GPUJPEG)
and [nvTIFF](https://developer.nvidia.com/nvtiff-downloads).
[nvCOMP](https://developer.nvidia.com/nvcomp) is also required for
DEFLATE decompress. **PNG** and **WebP** outputs are also supported
(although using CPU encoders).

If nvTIFF isn't capable to decompress the picture _libtiff_ is used
instead.

# Table of Contents

- [NEWS](#news)
  * [2026-02](#2026-02)
  * [Older](#older)
- [Requirements](#requirements)
  * [Software](#software)
  * [Hardware](#hardware)
- [Build and running](#build-and-running)
- [Usage](#usage)
  * [Combined features - RGB mapped](#combined-features---rgb-mapped)
  * [Combined features - normalized differential](#combined-features---normalized-differential)
- [Supported input formats](#supported-input-formats)
- [Performance](#performance)
  * [Output format comparison](#output-format-comparison)
- [TODO](#todo)
- [Postprocess - output normalization/equalization](#postprocess---output-normalizationequalization)
- [Notes and issues](#notes-and-issues)
- [Sample images' notes](#sample-images-notes)

<!-- gen with https://github.com/jonschlinkert/markdown-toc -->

# NEWS

## 2026-02

- [S2] add some normalized-differential features - HONS, NDVI, NDWI, NDMI, NDSI
- for S2 features, the source areas without significant data (NODATA)
are set transparent if WebP is output (alternatively in case of JPEG,
it will be colored according to the user-selected fill-color)
- **changed** the equalization of S2 features from _auto-equalize_
(still used for S1) to the defined range (valid range 1000-11000 for
B bands as defined by the MTD_MSIL2A.xml - _BOA_ADD_OFFSET_=-1000 and
_BOA_QUANTIFICATION_VALUE_=10000; the later value is mentioned also in
<https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/>)
- **changed** gamma-correction is no longer applied to S2 to get
cannonical representation (see previous point)
- **changed** gamma-correction is no longer applied to S1, neither,
after the auto-equalization - the decision to use gamma-correction is
somehow arbitrary so it was changed to align with S2
- **added** possibility for different tiles and whole image format
(arbitrary combination, see help)
- **added** PNG encoder (using fpnge)

## Older

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

Supported features are **HONS¹**, **NDVI²**, **NDWI**,
**NDMI**, **NDSI** and generic **ND_UNSPEC** (rendered as a
grayscale).  The list of required input bands follows, the arg syntax
is `feature@fname1,fname2` where  the feature name is one of the
above. Output file name is `feature-oname1-COMMA-oname2.ext` where
the oname is specified in previous section (**note** between _feature_
and _oname1_ is '-' not at-sign).

Required bands for features:
- **HONS¹** (_Highlight Optimized Natural Color_): _B04_ (red), _B03_ (green), _B02_ (blue)
- **NDVI²**: _B08_ (NIR - near infra red), _B04_ (red)
- **NDWI**: _B03_ (green), _B08_ (NIR)
- **NDMI**: _B8A_ (Narrow NIR), _B11_ (SWIR - short wave infrared)
- **NDSI**: _B03_ (green), _B11_ (SWIR), _B04_ (red) , _B02_ (blue)
- **ND_GENERIC**: any1, any2

Missing:
- **Scene classification map** - requires perhaps _SCL_ [L2A] band, which I don't have

Notes:

1. the output differs from Copernicus browser but when using the formula:

   ```
   return [Math.cbrt(0.6*B04 - 0.035),
           Math.cbrt(0.6*B03 - 0.035),
           Math.cbrt(0.6*B02 - 0.035)]
   ```
   literallly:
   [L1C Math.cbrt(0.6 * sample.B0X - 0.035) formula](https://browser.dataspace.copernicus.eu/?zoom=13&lat=48.89548&lng=16.663&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX18S0RU3ThQ5S1NI8XgRpm%2BIRCtsv%2FYiPKPMfauXlTOD0N%2FUzNe%2Bm60vDu4LEPZ%2F8KmzgickbVvzr%2F2sgbQJ116mzufvaf8P7POn8caCfyTmwJv9%2B12IJZRn&evalscript=Ly9WRVJTSU9OPTMKZnVuY3Rpb24gc2V0dXAoKSB7CiAgcmV0dXJuIHsKICAgIGlucHV0OiBbIkIwNCIsIkIwMyIsIkIwMiJdLAogICAgb3V0cHV0OiB7IGJhbmRzOiAzIH0KICB9Owp9CgpmdW5jdGlvbiBldmFsdWF0ZVBpeGVsKHNhbXBsZSkgewogIHJldHVybiBbTWF0aC5jYnJ0KDAuNiAqIHNhbXBsZS5CMDQgLSAwLjAzNSksCiAgICAgICAgTWF0aC5jYnJ0KDAuNiAqIHNhbXBsZS5CMDMgLSAwLjAzNSksCiAgICAgICAgTWF0aC5jYnJ0KDAuNiAqIHNhbXBsZS5CMDIgLSAwLjAzNSldCn0%3D&datasetId=S2_L1C_CDAS&fromTime=2025-11-04T00%3A00%3A00.000Z&toTime=2025-11-04T23%3A59%3A59.999Z&handlePositions=0%2C1&gradient=0x000000%2C0xffffff&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE#custom-script)

   we got the exactly same results but the reference:
   [L1C HONS](https://browser.dataspace.copernicus.eu/?zoom=13&lat=48.89548&lng=16.663&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX197JuTNIb0OFC0iBRgrbH3B%2BfOVyuMETBegSUmTYfepKBxy0JXjoN4e2Q3%2BJReC%2BHgdIBOYFsFlmLFRqw%2Bg%2BoA7qa8CgCuSEZ9U%2Fgagt4fGf%2BLMFIKi48s6&datasetId=S2_L1C_CDAS&fromTime=2025-11-04T00%3A00%3A00.000Z&toTime=2025-11-04T23%3A59%3A59.999Z&layerId=2_TONEMAPPED_NATURAL_COLOR&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE)
   differs.

2. the ramp for **NDVI** used by _gjtiff_ differs from the one
described [here](https://web.archive.org/web/20251108154243/https://custom-scripts.sentinel-hub.com/sentinel-2/ndvi/)

   It seems like the Copernicus browser isn't using that exactly, eg:
   [L2A (B08-B04)/(B08+B04) as grayscale with -0.5 threshold](https://browser.dataspace.copernicus.eu/?zoom=13&lat=48.88351&lng=16.62918&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2Fs1%2B6lSY5DGIzAzp0k9IuA2HWSsjBYAzM2HhVye%2BeoO%2Fjzb4iaqY5zqA5CL5%2BDuE6w%2BdQ8K3ya%2BWtf21PNhkeey7jW0W5NQUfhNJNNOsfo%2BwUo1vnFwRsD&evalscript=Ly9WRVJTSU9OPTMKZnVuY3Rpb24gc2V0dXAoKSB7CiAgcmV0dXJuIHsKICAgIGlucHV0OiBbIkIwNCIsIkIwOCIsIkIxMSIsICJCOEEiXSwKICAgIG91dHB1dDogeyBiYW5kczogNCB9CiAgfTsKfQoKZnVuY3Rpb24gZXZhbHVhdGVQaXhlbChzYW1wbGUpIHsKICB2YWw9KHNhbXBsZS5CMDgtc2FtcGxlLkIwNCkvKHNhbXBsZS5CMDgrc2FtcGxlLkIwNCkvMiArIDAuNQogIHZhbCA9IHZhbCA8IC4yNSA%2FIDAgOiAxCgogIHJldHVybiBbdmFsLCB2YWwsdmFsICwgMV07Cn0%3D&datasetId=S2_L2A_CDAS&fromTime=2025-11-04T00%3A00%3A00.000Z&toTime=2025-11-04T23%3A59%3A59.999Z&handlePositions=0%2C1&gradient=0x000000%2C0xffffff&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE#custom-script)
   shows that almost all area of Novomlýnské nádrže has the value <0.25, which
   is < -0.5 if range is [-1,1], so that it should be all black according to the scale
   but it isn't in the reference NDVI render:
   [L2A NDVI reference](https://browser.dataspace.copernicus.eu/?zoom=13&lat=48.88351&lng=16.62918&themeId=DEFAULT-THEME&visualizationUrl=U2FsdGVkX1%2F9BT1QzDTrEyynQPU1mLZ92ge5e7gGxYW47%2F5lpyVIYO42IzCwh7cFEiXO5r4VkCtck4%2BKgUWkHtbp9v8%2BjbAjeQvzot6QRBvvwoU1S%2FC7%2BXerYW6lej1U&datasetId=S2_L2A_CDAS&fromTime=2025-11-04T00%3A00%3A00.000Z&toTime=2025-11-04T23%3A59%3A59.999Z&layerId=3_NDVI&demSource3D=%22MAPZEN%22&cloudCoverage=30&dateMode=SINGLE)

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

## Output format comparison

Used images:
- **T33UXQ_20251104T095211_B03_10m.jp2** - green band, _grayscale output_ (with alpha)
- **HONS@T33UXQ_20251104T095211_B04,T33UXQ_20251104T095211_B03,T33UXQ_20251104T095211_B02** - Highlight Optimized Natural Color, _RGBA output_

| config               | encode (s) | duration (altogether) | size (MB, excluding tiles**¹**) |
| -------------------- | ---------- | --------------------- | ------------------------------- |
| **PNG**              |            |                       |                                 |
| -p green             | 0.13       | 1.03                  |  61.91                          |
| -p -z 15 green       | 1.14       | 2.01                  |  61.91                          |
| -p HONS              | 0.33       | 1.77                  | 181.01                          |
| -p -z 15 HONS        | 2.19       | 3.69                  | 181.01                          |
| **WebP**             |            |                       |                                 |
| -w -q 82 green       | 0.78       | 1.70                  |   4.91                          |
| -w -q 82 -z 15 green | 2.00       | 2.85                  |   4.91                          |
| -w -q 82 HONS        | 1.31       | 2.72                  |  16.33                          |
| -w -q 82 -z 15 HONS  | 2.46       | 3.98                  |  16.33                          |


**¹** for zoom-level=15, the tiles take approximately 7x more space than the image itself

**PNG** is using [fpnge](https://github.com/veluca93/fpnge) encoder,
[fpng](https://github.com/richgel999/fpng) can be also used by setting
environment variable _PNG_BACKEND=fpng_.

<details>
<summary>PNG results with fpng backend</summary>
<table>
<tr><th> config               <th> encode (s) <th> duration (altogether) <th> size (MB, excluding tiles)
<tbody>
<tr><td> -p green             <td> 0.71       <td> 1.54                  <td> 152.22
<tr><td> -p -z 15 green       <td> 2.18       <td> 3.06                  <td> 152.22
<tr><td> -p HONS              <td> 0.64       <td> 2.17                  <td> 205.68
<tr><td> -p -z 15 HONS        <td> 2.17       <td> 3.78                  <td> 205.68
</table>
</details>

As **PNG** encoders, _stb_image_writer_ and libpng were also considered, especially
for the grayscale+alpha case. But in the best settings were slower, eg. for the green
libpng took 2.5 sec and stb 3.5.

The whole image and individual tiles are compressed in parallel. Basically,
if using tiles, the encode time is the duration of encoding tiles for z=15.
The compression of the whole and tiles takes approximately the same at z=14
using the AMD Ryzen 9 7900X processor (so that at lower zoom level, the
whole image will become the critical path).

# TODO

- improved nvjpeg2000 decoding performance if needed
<https://developer.nvidia.com/blog/accelerating-jpeg-2000-decoding-for-digital-pathology-and-satellite-images-using-the-nvjpeg2000-library/>
- various performance optimizations - eg. when image sizes change
often, which is slow due to GPUJPEG reconfiguration
- cudaMalloc/Free Async version should be used when used within
streams (as deployed now, it doesn't cause probbems right now)

# Postprocess - output normalization/equalization

The **S1/tiff** images are converted to visual representation by
equalizing the input as present in _SNAP_ <del>and applying **gamma=2**
(not present in SNAP)</del>. The
equalization is performed to scale input [0, µ + 2σ] to output [0,
255]. If maximal sample in the set is less than µ  + 2σ, this is used
instead (not likely but happens sometimes when the input values are very
small). The value above are clamped to 255.

**Sentinel-2 JP2** bands are equalized from range 1000-11000
(_BOA_ADD_OFFSET_=1000 and _BOA_QUANTIFICATION_VALUE_=10000 in
[MTD_MSIL2A.xml](doc/MTD_MSIL2A.xml), the later value also
[here](https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/)).

For **3-channel J2K images** (_S2 TCI_) the normalization (neither
Gamma-compression)
doesn't take place because those seem to represent visual-range data
(including the Gamma). Note that SNAP handles the image bands separately
(as a grayscale) and it also undergoes the equalization (by default).


# Notes and issues

- NDVI rendering doesn't use reference scale (see the '@^{/fix NDVI}' commit)

# Sample images' notes

- Copernicus/download/Landsat-8/OLI_TIRS/L1TP semm to be in visible range
- Sentinel-1 need to have the levels adjusted
- Sentinel-2 in J2K (the support needs to be added)

# References:
[L1C]: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/
[L2A]: https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l2a/

- [L1C]
- [L2A]
