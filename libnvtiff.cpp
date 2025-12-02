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

#include "libnvtiff.h"

#include <cstdint>
#include <cinttypes>       // for PRIu32
#include <cuda_runtime.h>  // for cudaFree, cudaMalloc, cudaStreamDestroy
#include <nvtiff.h>        // for NVTIFF_STATUS_SUCCESS, nvtiffStatus_t, nvt...
#include <nvtiff_version.h> // for NVTIFF_VER_MAJOR, NVTIFF_VER_MINOR
#include <stdint.h>        // for uint8_t, uint16_t, uint32_t
#include <stdlib.h>        // for exit, EXIT_FAILURE
#include <cassert>         // for assert
#include <cstdio>          // for fprintf, stderr, size_t

#include "defs.h"          // for dec_image, CHECK_CUDA, rc
#include "gdal_coords.h"   // for set_suggested_size_from_gdal
#include "kernels.h"     // for convert_16_8_normalize_cuda
#include "libtiffinfo.hpp" // for tiff_get_corners
#include "utils.h"         // for ERROR_MSG

#define DIV_UP(a, b) (((a) + ((b) - 1)) / (b))

#define CHECK_NVTIFF(call)                                                     \
        {                                                                      \
                nvtiffStatus_t _e = (call);                                    \
                if (_e != NVTIFF_STATUS_SUCCESS) {                             \
                        fprintf(stderr,                                        \
                                "nvtiff error code %d (%s) in file '%s' in "   \
                                "line %i\n",                                   \
                                _e, nvtiff_status_to_str(_e), __FILE__,        \
                                __LINE__);                                     \
                        exit(EXIT_FAILURE);                                    \
                }                                                              \
        }

struct nvtiff_state {
        nvtiff_state(cudaStream_t cs, int ll);
        ~nvtiff_state();
        int log_level;
        nvtiffStream_t tiff_stream{};
        nvtiffDecoder_t decoder{};
        cudaStream_t stream{};
        uint8_t *decoded{};
        size_t decoded_allocated{};
        // // converted
        uint8_t *converted{};
        size_t converted_allocated{};

#if NVTIFF_VER_MINOR > 0 || NVTIFF_VER_MINOR >= 4
        nvtiffDecodeParams_t params;
#endif

        void *tiff_info_buf{};
        size_t tiff_info_buf_sz = 0;
};

nvtiff_state::nvtiff_state(cudaStream_t cs, int ll)
    : log_level(ll), stream(cs)
{
        CHECK_NVTIFF(nvtiffStreamCreate(&tiff_stream));
        CHECK_NVTIFF(nvtiffDecoderCreateSimple(&decoder, stream));
#if NVTIFF_VER_MINOR > 0 || NVTIFF_VER_MINOR >= 4
        CHECK_NVTIFF(nvtiffDecodeParamsCreate(&params));
#endif
}

nvtiff_state::~nvtiff_state()
{
        CHECK_NVTIFF(nvtiffStreamDestroy(tiff_stream));
        CHECK_NVTIFF(nvtiffDecoderDestroy(decoder, stream));
#if NVTIFF_VER_MINOR > 0 || NVTIFF_VER_MINOR >= 4
        CHECK_NVTIFF(nvtiffDecodeParamsDestroy(params));
#endif
        CHECK_CUDA(cudaFree(decoded));
        CHECK_CUDA(cudaFree(converted));
}

struct nvtiff_state *nvtiff_init(cudaStream_t stream, int log_level) {
        return new nvtiff_state(stream, log_level);
}

void nvtiff_destroy(struct nvtiff_state *s)
{
        delete s;
}

static uint8_t *convert_16_8(struct nvtiff_state  *s, struct dec_image *img, float *scale)
{
        const size_t out_size = (size_t) img->width * img->height;
        if (out_size > s->converted_allocated) {
                CHECK_CUDA(cudaFree(s->converted));
                CHECK_CUDA(cudaMalloc(&s->converted, out_size));
                s->converted_allocated = out_size;
        }
        *scale = convert_16_8_normalize_cuda(img, s->converted, s->stream);
        return s->converted;
}

/// check file emptiness - nvtiff calls exit() for emtpy files so get around
/// this
static bool is_empty(const char *fname)
{
        FILE *f = fopen(fname, "rb");
        if (f == nullptr) {
                perror("fopen");
                return false;
        }
        fseek(f, 0, SEEK_END);
        long off = ftell(f);
        fclose(f);
        return off == 0;
}

static void set_coords_from_geotiff(struct nvtiff_state *s, uint32_t image_id,
                                    struct dec_image *image)
{
        image->coords_set = false;

        nvtiffTagDataType_t tag_type{};
        uint32_t size{};
        uint32_t count{};
        if (NVTIFF_STATUS_SUCCESS !=
            nvtiffStreamGetTagInfo(s->tiff_stream, image_id,
                                   NVTIFF_TAG_MODEL_TIE_POINT, &tag_type, &size,
                                   &count)) {
                WARN_MSG("Image coordintates cannot be set - tag not found.\n");
                return;
        }
        if (tag_type != NVTIFF_TAG_TYPE_DOUBLE) {
                WARN_MSG("Image coordintates cannot be set - type %d is not "
                         "double\n",
                         (int)tag_type);
                return;
        }
        if (count % 6 != 0) {
                WARN_MSG(
                    "Image coordintates points is not a 6-tuple (mod 6)!\n");
                return;
        }

        const size_t required_sz = (size_t)size * count;

        if (required_sz >= s->tiff_info_buf_sz) {
                s->tiff_info_buf_sz = required_sz;
                s->tiff_info_buf = realloc(s->tiff_info_buf,
                                           s->tiff_info_buf_sz);
                assert(s->tiff_info_buf != nullptr);
        }
        CHECK_NVTIFF(nvtiffStreamGetTagValue(s->tiff_stream, image_id,
                                             NVTIFF_TAG_MODEL_TIE_POINT,
                                             s->tiff_info_buf, count));
        assert(count % 6 == 0);
        image->tie_points = tuple6_to_tie_points(count / 6,
                                                 (double *)s->tiff_info_buf);

        if (tiff_get_corners_bounds(image->tie_points, image->width,
                                    image->height, image->coords,
                                    image->bounds)) {
                image->coords_set = true;
        }
}

#if NVTIFF_VER_MINOR > 0 || NVTIFF_VER_MINOR >= 4
static void print_tiff_info(nvtiffStream_t tiff_stream, uint32_t image_id)
{
        if (log_level < LL_DEBUG) {
                return;
        }
        nvtiffImageGeometry_t geometry;
        if (nvtiffStreamGetImageGeometry(tiff_stream, image_id, &geometry) !=
            NVTIFF_STATUS_SUCCESS) {
                return;
        }
        DEBUG_MSG("TIFF structure:\n");
        DEBUG_MSG("\ttype: %s\n",
               geometry.type == NVTIFF_IMAGE_STRIPED ? "stripped" : "tiled");
        const char *strile_type = geometry.type == NVTIFF_IMAGE_STRIPED
                                      ? "stripe"
                                      : "tile";
        DEBUG_MSG("\tdepth: %" PRIu32 "\n", geometry.image_depth);
        DEBUG_MSG("\t%s width: %" PRIu32 "\n", strile_type,
                  geometry.strile_width);
        DEBUG_MSG("\t%s height: %" PRIu32 "\n", strile_type,
                  geometry.strile_height);
        DEBUG_MSG("\t%s depth: %" PRIu32 "\n", strile_type,
                  geometry.strile_depth);
        DEBUG_MSG("\tnum %ss per plane: %" PRIu32 "\n", strile_type,
                  geometry.num_striles_per_plane);
        DEBUG_MSG("\tnum %ss: %" PRIu32 "\n", strile_type,
                  geometry.num_striles);
}
#endif

/**
 * Decodes TIFF using nvTIFF.
 *
 * If DEFLATE-compressed TIFF is detected but nvCOMP not found in
 * library lookup path (LD_LIBRARY_PATH on Linux), exit() is called
 * unless enforced use of libtiff.
 */
struct dec_image nvtiff_decode(struct nvtiff_state *s, const char *fname)
{
        struct dec_image ret{};
        if (is_empty(fname)) {
               ERROR_MSG("%s is empty!\n", fname);
               return {};
        }
        GPU_TIMER_START(nvtiff_decode, LL_DEBUG, s->stream);
        // CHECK_NVTIFF(nvtiffStreamGetNumImages(tiff_stream, &num_images));
        nvtiffStatus_t e = nvtiffStreamParseFromFile(fname, s->tiff_stream);
        if (e == NVTIFF_STATUS_TIFF_NOT_SUPPORTED) {
                WARN_MSG("%s not supported by nvtiff\n", fname);
                return {};
        }

#define CHECK_ERROR                                                            \
        if (e != NVTIFF_STATUS_SUCCESS) {                                      \
                ERROR_MSG("nvtiff error code %d in file '%s' in line %i for "  \
                          "%s: %s\n",                                          \
                          e, __FILE__, __LINE__, fname,                        \
                          nvtiff_status_to_str(e));                            \
                return {};                                                     \
        }

        CHECK_ERROR

        if (s->log_level >= 2) {
                CHECK_NVTIFF(nvtiffStreamPrint(s->tiff_stream));
        }
        const int image_id = 0; // first image only
        nvtiffImageInfo_t image_info{};
        CHECK_NVTIFF(
            nvtiffStreamGetImageInfo(s->tiff_stream, image_id, &image_info));
        assert(image_info.photometric_int != NVTIFF_PHOTOMETRIC_PALETTE);
        const size_t nvtiff_out_size =
            DIV_UP((size_t)image_info.bits_per_pixel * image_info.image_width,
                   8) *
            (size_t)image_info.image_height;
        if (nvtiff_out_size > s->decoded_allocated) {
                CHECK_CUDA(cudaFree(s->decoded));
                CHECK_CUDA(cudaMalloc(&s->decoded, nvtiff_out_size));
                s->decoded_allocated = nvtiff_out_size;
        }
#if NVTIFF_VER_MINOR > 0 || NVTIFF_VER_MINOR >= 4
        print_tiff_info(s->tiff_stream, image_id);
        e = nvtiffDecodeImage(s->tiff_stream, s->decoder, s->params, image_id,
                              s->decoded, s->stream);
#else
        const uint32_t num_images = 1;
        e = nvtiffDecodeRange(s->tiff_stream, s->decoder, image_id, num_images,
                              &s->decoded, s->stream);
#endif
        if (e == NVTIFF_STATUS_NVCOMP_NOT_FOUND) {
                ERROR_MSG("nvCOMP needed for DEFLATE not found in path...\n");
                ret.rc = ERR_NVCOMP_NOT_FOUND;
                return ret;
        }
        CHECK_ERROR
        ret.width = image_info.image_width;
        ret.height = image_info.image_height;
        ret.comp_count = image_info.samples_per_pixel;
        ret.data = s->decoded;
        set_coords_from_geotiff(s, image_id, &ret);
        set_suggested_size_from_gdal(fname, &ret);
        if (image_info.bits_per_sample[0] == 8) {
                return ret;
        }
        assert(image_info.bits_per_sample[0] == 16);
        ret.data = convert_16_8(s, &ret, &ret.scale);
        GPU_TIMER_STOP(nvtiff_decode);
        return ret;
}
