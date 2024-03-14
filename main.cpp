/**
 * @file
 * For reference see:
 * <https://github.com/NVIDIA/CUDALibrarySamples/blob/9d3da0b1823d679ff0cc8e9ef05c6cc93dc5ad76/nvTIFF/nvTIFF-Decode-Encode/nvtiff_example.cu>
 */
/*
 * Copyright (c) 2024,       CESNET, z. s. p. o.
 * Copyright (c) 2022 -2023, NVIDIA CORPORATION. All rights reserved.
 *
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <libgpujpeg/gpujpeg_type.h>
#include <nvtiff.h>

#include "libtiff.hpp"
#include "kernels.hpp"

#define DIV_UP(a, b) (((a) + ((b)-1)) / (b))

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CHECK_NVTIFF(call)                                                     \
  {                                                                            \
    nvtiffStatus_t _e = (call);                                                \
    if (_e != NVTIFF_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "nvtiff error code %d in file '%s' in line %i\n", _e,    \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

struct state_gjtiff {
  state_gjtiff(int log_level);
  ~state_gjtiff();
  int log_level;
  // NVTIFF
  nvtiffStream_t tiff_stream{};
  nvtiffDecoder_t decoder{};
  cudaStream_t stream{};
  uint8_t *decoded{};
  size_t decoded_allocated{};
  // libtiff
  libtiff_state tiff_state;
  // converted
  uint8_t *converted{};
  size_t converted_allocated{};
  // GPUJPEG
  struct gpujpeg_encoder *gj_enc{};
};

state_gjtiff::state_gjtiff(int l) : log_level(l), tiff_state(l) {
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
  CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
  gj_enc = gpujpeg_encoder_create(stream);
  assert(gj_enc != nullptr);
}

state_gjtiff::~state_gjtiff() {
  CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
  CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFree(decoded));
  CHECK_CUDA(cudaFree(converted));
  gpujpeg_encoder_destroy(gj_enc);
}

static uint8_t *decode_tiff(struct state_gjtiff *s, const char *fname,
                            size_t *nvtiff_out_size,
                            nvtiffImageInfo_t *image_info) {
  printf("Decoding from file %s... \n", fname);
  const uint32_t num_images = 1;
  // CHECK_NVTIFF(nvtiffStreamGetNumImages(tiff_stream, &num_images));
  nvtiffStatus_t e = nvtiffStreamParseFromFile(fname, s->tiff_stream);
  if (e == NVTIFF_STATUS_TIFF_NOT_SUPPORTED) {
    fprintf(stderr, "TIFF not supported by nvtiff, trying libtiff...\n");
    return s->tiff_state.decode(fname, nvtiff_out_size, image_info,
                                   &s->decoded, &s->decoded_allocated, s->stream);
  } else if (e != NVTIFF_STATUS_SUCCESS) {
    fprintf(stderr, "nvtiff error code %d in file '%s' in line %i\n", e,
            __FILE__, __LINE__);
    return nullptr;
  }
  if (s->log_level >= 2) {
    CHECK_NVTIFF(nvtiffStreamPrint(s->tiff_stream));
  }
  const int image_id = 0; // first image only
  CHECK_NVTIFF(
      nvtiffStreamGetImageInfo(s->tiff_stream, image_id, image_info));
  assert(image_info->photometric_int != NVTIFF_PHOTOMETRIC_PALETTE);
  *nvtiff_out_size =
      DIV_UP((size_t)image_info->bits_per_pixel * image_info->image_width, 8) *
      (size_t)image_info->image_height;
  if (*nvtiff_out_size > s->decoded_allocated) {
    CHECK_CUDA(cudaFree(s->decoded));
    CHECK_CUDA(cudaMalloc(&s->decoded, *nvtiff_out_size));
    s->decoded_allocated = *nvtiff_out_size;
  }
  CHECK_NVTIFF(nvtiffDecodeRange(s->tiff_stream, s->decoder, image_id,
                                 num_images, &s->decoded, s->stream));
  return s->decoded;
}

static uint8_t *convert_16_8(struct state_gjtiff *s, uint8_t *in, int in_depth,
                             size_t in_size) {
  if (in_depth == 8) {
    return in;
  }
  assert(in_depth == 16);
  const size_t out_size = in_size / 2;
  if (out_size > s->converted_allocated) {
    CHECK_CUDA(cudaFree(s->converted));
    CHECK_CUDA(cudaMalloc(&s->converted, out_size));
    s->converted_allocated = out_size;
  }
  convert_16_8_cuda((uint16_t *) in, s->converted, in_size, s->stream);
  return s->converted;
}

static void encode_jpeg(struct state_gjtiff *s, uint8_t *cuda_image,
                        int comp_count, int width, int height,
                        const char *ofname) {
  gpujpeg_parameters param;
  gpujpeg_set_default_parameters(&param);

  gpujpeg_image_parameters param_image;
  gpujpeg_image_set_default_parameters(&param_image);
  param_image.width = width;
  param_image.height = height;
  param_image.comp_count = comp_count;
  param_image.color_space = comp_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
  param_image.pixel_format = comp_count == 1 ? GPUJPEG_U8 : GPUJPEG_444_U8_P012;
  gpujpeg_encoder_input encoder_input;
  gpujpeg_encoder_input_set_gpu_image(&encoder_input, cuda_image);
  uint8_t *out = nullptr;
  size_t len = 0;
  gpujpeg_encoder_encode(s->gj_enc, &param, &param_image, &encoder_input, &out,
                         &len);

  FILE *outf = fopen(ofname, "wb");
  if (outf == nullptr) {
    fprintf(stderr, "fopen %s: %s\n", ofname, strerror(errno));
    return;
  }
  fwrite(out, len, 1, outf);
  fclose(outf);
}

static void set_ofname(const char *ifname, char *ofname, size_t buflen) {
  if (strrchr(ifname, '/') != nullptr) {
    snprintf(ofname, buflen, "%s", strrchr(ifname, '/') + 1);
  } else {
    snprintf(ofname, buflen, "%s", ifname);
  }
  if (strrchr(ofname, '.') != nullptr) {
    char *ptr = strrchr(ofname, '.') + 1;
    size_t avail_len = buflen - (ptr - ofname);
    snprintf(ptr, avail_len, "jpg");
  } else {
    snprintf(ofname + strlen(ofname), buflen - strlen(ofname), ".jpg");
  }
}

static void show_help(const char *progname) {
  printf("%s [-o <outdir>] [-v[v]] img1.tif [img2.tif...]\n\n", progname);
  printf("Output filename will be \"basename ${name%%.*}.jpg\"\n");
  printf("Output directory must exist, implicitly \".\"\n");
}

int main(int /* argc */, char **argv) {
  int log_level = 0;
  char ofname[1024] = "./";
  const char *progname = argv[0];

  argv++;
  while (*argv != nullptr && argv[0][0] == '-') {
    argv++;
    if (strcmp(argv[-1], "--") == 0) {
      break;
    }
    if (strcmp(argv[-1], "-h") == 0) {
      show_help(progname);
      return 0;
    }
    if (strncmp(argv[-1], "-o", 2) == 0) {
      if (strlen(argv[-1]) > 2) { // -o<dir>
        snprintf(ofname, sizeof ofname, "%s/", argv[-1] + 2);
      } else { // -o <dir>
        assert(argv[0] != nullptr);
        snprintf(ofname, sizeof ofname, "%s/", argv[0]);
        argv++;
      }
    } else if (strstr(argv[-1], "-v") == argv[-1]) {
      log_level += std::count(argv[-1], argv[-1] + strlen(argv[-1]), 'v');
    } else {
      fprintf(stderr, "Unknown option: %s!\n", argv[-1]);
      return 1;
    }
  }

  struct state_gjtiff s(log_level);

  const size_t d_pref_len = strlen(ofname);
  for (; *argv != nullptr; ++argv) {
    const char *ifname = argv[0];
    set_ofname(ifname, ofname + d_pref_len, sizeof ofname - d_pref_len);

    nvtiffImageInfo_t image_info;
    size_t nvtiff_out_size;
    uint8_t *decoded = decode_tiff(&s, ifname, &nvtiff_out_size, &image_info);
    if (decoded == nullptr) {
      continue;
    }
    uint8_t *converted = convert_16_8(
        &s, decoded, image_info.bits_per_sample[0], nvtiff_out_size);
    encode_jpeg(&s, converted, image_info.samples_per_pixel,
                image_info.image_width, image_info.image_height, ofname);
  }
}
