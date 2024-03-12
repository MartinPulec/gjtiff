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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <libgpujpeg/gpujpeg_type.h>
#include <nvtiff.h>
#include <vector>

#include "kernels.hpp"

using std::vector;

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

static void encode_jpeg(uint8_t *cuda_image, int comp_count, int width, int height) {
  auto *gj_enc = gpujpeg_encoder_create(0);
  assert(gj_enc != NULL);
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
  gpujpeg_encoder_encode(gj_enc, &param, &param_image, &encoder_input, &out,
                         &len);

  FILE *outf = fopen("out.jpg", "wb");
  fwrite(out, len, 1, outf);
  fclose(outf);

  gpujpeg_encoder_destroy(gj_enc);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  const char *fname = argv[1];
  const int verbose = 1;
  int frameBeg = INT_MIN;
	int frameEnd = INT_MAX;
  nvtiffStream_t tiff_stream;
  nvtiffDecoder_t decoder;
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
  CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
  CHECK_NVTIFF(nvtiffStreamParseFromFile(fname, tiff_stream));
  uint32_t num_images = 0;
  CHECK_NVTIFF(nvtiffStreamGetNumImages(tiff_stream, &num_images));
  num_images = 1; // only one first tile for now
  vector<nvtiffImageInfo_t> image_info(num_images);
  vector<uint8_t *> nvtiff_out(num_images);
  vector<size_t> nvtiff_out_size(num_images);

  // BEGIN work (possibly) overlapped with H2D copy of the file data
	if (verbose) {
		CHECK_NVTIFF(nvtiffStreamPrint(tiff_stream));
	}
	
	frameBeg = fmax(frameBeg, 0);
	frameEnd = fmin(frameEnd, num_images-1);
	const int nDecode = frameEnd-frameBeg+1;

	for (uint32_t image_id = 0; image_id < num_images; image_id++) {
        CHECK_NVTIFF(nvtiffStreamGetImageInfo(tiff_stream, image_id, &image_info[image_id]));
        nvtiff_out_size[image_id] = DIV_UP((size_t)image_info[image_id].bits_per_pixel * image_info[image_id].image_width, 8) *
                                    (size_t)image_info[image_id].image_height;
        if (image_info[image_id].photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
            nvtiff_out_size[image_id] = image_info[image_id].image_width * image_info[image_id].image_height * 3 * sizeof(uint16_t);
        }
        CHECK_CUDA(cudaMalloc(&nvtiff_out[image_id], nvtiff_out_size[image_id]));
    }

	printf("Decoding %u, images [%d, %d], from file %s... \n",
		nDecode,
		frameBeg,
		frameEnd,
		fname);
	fflush(stdout);

  CHECK_NVTIFF(nvtiffDecodeRange(tiff_stream, decoder, frameBeg, num_images, nvtiff_out.data(), stream));

  uint8_t *in_8 = nvtiff_out[0];
  uint8_t *tmpbuf = nullptr;
  if (image_info[0].bits_per_pixel == 16) {
    in_8 = tmpbuf = convert16_8((uint16_t *) nvtiff_out[0], nvtiff_out_size[0], stream);
  }

  encode_jpeg(in_8, image_info[0].samples_per_pixel,
              image_info[0].image_width, image_info[0].image_height);

  // cudaStreamSynchronize(stream);
  for (unsigned int i = 0; i < num_images; i++) {
    CHECK_CUDA(cudaFree(nvtiff_out[i]));
  }

  CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
  CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
}
