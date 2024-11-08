#include "libnvj2k.h"

#include <cerrno>              // for errno
#include <cinttypes>           // for PRIu32, PRIu8
#include <cstdint>             // for uint8_t
#include <cstdio>              // for fseek, fclose, fopen, fread, ftell, FILE
#include <cstdlib>             // for calloc, free
#include <cstring>             // for size_t, strerror
#include <ctime>               // for timespec
#include <cuda_runtime_api.h>  // for cudaFree, cudaFreeHost, cudaMalloc
#include <nvjpeg2k.h>

#include "defs.h"              // for dec_image, log_level, rc
#include "kernels.hpp"         // for convert_planar_rgb_to_packed
#include "utils.hpp"           // for ERROR_MSG, TIMER_START, TIMER_STOP

enum {
        MAX_COMPONENTS = 3,
};

struct nvj2k_state {
        nvjpeg2kHandle_t nvjpeg2k_handle;
        nvjpeg2kStream_t nvjpeg2k_stream;
        nvjpeg2kDecodeParams_t decode_params;
        nvjpeg2kDecodeState_t decode_state;

        cudaStream_t cuda_stream;

        char *bitstream_buffer;
        size_t bitstream_buffer_allocated;

        uint8_t *decode_output;
        unsigned dec_out_linesize_allocated,
            dec_out_height_allocated;
        size_t pitch_in_bytes;

        uint8_t *converted;
        size_t converted_allocated;
};

// prototypes
static void print_j2k_info(const nvjpeg2kImageInfo_t *image_info,
                           const nvjpeg2kImageComponentInfo_t *image_comp_info);
static bool
validate_equal_comps(const nvjpeg2kImageInfo_t *image_info,
                     const nvjpeg2kImageComponentInfo_t *image_comp_info);

/**
 * @todo error handling
 */
struct nvj2k_state *nvj2k_init(cudaStream_t stream) {
        struct nvj2k_state *s = (struct nvj2k_state *) calloc(1, sizeof *s);

        nvjpeg2kCreateSimple(&s->nvjpeg2k_handle);
        nvjpeg2kDecodeStateCreate(s->nvjpeg2k_handle, &s->decode_state);
        nvjpeg2kStreamCreate(&s->nvjpeg2k_stream);
        nvjpeg2kDecodeParamsCreate(&s->decode_params);
        nvjpeg2kDecodeParamsSetOutputFormat(s->decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
        s->cuda_stream = stream;
        return s;
}

struct dec_image nvj2k_decode(struct nvj2k_state *s, const char *fname) {
        FILE *f = fopen(fname, "rb");
        if (f == nullptr) {
                ERROR_MSG("Unable to open file %s: %s\n", fname, strerror(errno));
                return {};
        }
        fseek(f, 0, SEEK_END);
        const long filesize = ftell(f);
        if (s->bitstream_buffer_allocated < (size_t) filesize) {
                cudaFreeHost(s->bitstream_buffer);
                cudaMallocHost((void **)&s->bitstream_buffer, filesize);
                s->bitstream_buffer_allocated = filesize;
        }
        fseek(f, 0, SEEK_SET);
        const size_t read_bytes = fread(s->bitstream_buffer, 1, filesize, f);
        fclose(f);
        if (read_bytes != (size_t) filesize) {
                ERROR_MSG("Read just %zu bytes from %s but the file has %ld "
                          "bytes!\ns\n",
                          read_bytes, fname, filesize);
                return {};
        }

        nvjpeg2kStatus_t status = nvjpeg2kStreamParse(
            s->nvjpeg2k_handle, (unsigned char *)s->bitstream_buffer, filesize,
            0, 0, s->nvjpeg2k_stream);
        if (status != NVJPEG2K_STATUS_SUCCESS) {
                ERROR_MSG("Unable to parse J2K bitstream for file %s: %d\n",
                          fname, (int)status);
                return {};
        }

        // extract image info
        nvjpeg2kImageInfo_t image_info;
        nvjpeg2kStreamGetImageInfo(s->nvjpeg2k_stream, &image_info);

        if (image_info.num_components > MAX_COMPONENTS) {
                ERROR_MSG("%s: assumed at most %d components, got %d!\n", fname,
                          (int)MAX_COMPONENTS, image_info.num_components);
                return {};
        }

        // assuming the decoding of images with 8 bit precision, and 3
        // components

        nvjpeg2kImageComponentInfo_t image_comp_info[MAX_COMPONENTS];

        for (unsigned c = 0; c < image_info.num_components; c++) {
                nvjpeg2kStreamGetImageComponentInfo(s->nvjpeg2k_stream,
                                                    &image_comp_info[c], c);
        }

        if (!validate_equal_comps(&image_info, image_comp_info)) {
                ERROR_MSG("%s: inequal components:\n", fname);
                print_j2k_info(&image_info, image_comp_info);
                return {};
        }
        if (image_comp_info[0].sgn != 0) {
                ERROR_MSG("%s: signed samples not yet supported!\n", fname);
                print_j2k_info(&image_info, image_comp_info);
        }
        if (log_level >= LL_DEBUG) {
                print_j2k_info(&image_info, image_comp_info);
        }

        nvjpeg2kImage_t output_image;
        unsigned bps = 1;
        output_image.pixel_type = NVJPEG2K_UINT8;
        output_image.num_components = image_info.num_components;
        if (image_comp_info[0].precision > 8) {
                bps = 2;
                output_image.pixel_type = NVJPEG2K_UINT16;
        }

        const unsigned linesize = image_comp_info[0].component_width * bps *
                                  image_info.num_components;
        if (linesize > s->dec_out_linesize_allocated ||
            image_comp_info[0].component_height > s->dec_out_height_allocated) {
                s->dec_out_linesize_allocated = MAX(
                    s->dec_out_linesize_allocated, linesize);
                s->dec_out_height_allocated = MAX(
                    s->dec_out_height_allocated,
                    image_comp_info[0].component_height);
                cudaFree(s->decode_output);
                cudaMallocPitch((void **)&s->decode_output, &s->pitch_in_bytes,
                                s->dec_out_linesize_allocated,
                                s->dec_out_height_allocated);
        }

        output_image.pixel_data = (void **) &s->decode_output;
        output_image.pitch_in_bytes = &s->pitch_in_bytes;

        TIMER_START(nvjpeg2kDecode, LL_DEBUG);
        status = nvjpeg2kDecodeImage(s->nvjpeg2k_handle, s->decode_state,
                                     s->nvjpeg2k_stream, s->decode_params,
                                     &output_image, s->cuda_stream);
        TIMER_STOP(nvjpeg2kDecode);
        if (status != NVJPEG2K_STATUS_SUCCESS) {
                ERROR_MSG("Unable to decode J2K file %s: %d\n", fname,
                          (int)status);
                return {};
        }

        size_t conv_size = (size_t)linesize *
                           image_comp_info[0].component_height;
        conv_size += conv_size / bps;
        if (s->converted_allocated < conv_size) {
                cudaFree(s->converted);
                cudaMalloc((void **)&s->converted, conv_size);
                s->converted_allocated = conv_size;
        }

        struct dec_image ret;
        ret.width = (int)image_comp_info[0].component_width;
        ret.height = (int)image_comp_info[0].component_height;
        ret.comp_count = (int) image_info.num_components;
        ret.data = s->converted;

        if (bps == 1) {
                convert_remove_pitch(
                    s->decode_output, s->converted,
                    (int)(image_comp_info[0].component_width * image_info.num_components),
                    (int)s->pitch_in_bytes, (int)image_comp_info[0].component_height,
                    s->cuda_stream);
                normalize_8(
                    &ret, s->converted + conv_size / 2, s->cuda_stream);
                ret.data = s->converted + conv_size / 2;
        } else {
                convert_remove_pitch_16(
                    (uint16_t *)s->decode_output,
                    (uint16_t *)s->converted,
                    (int)(image_comp_info[0].component_width * image_info.num_components),
                    (int)s->pitch_in_bytes, (int)image_comp_info[0].component_height,
                    s->cuda_stream);
                convert_16_8_normalize_cuda(
                    &ret, s->converted + conv_size / 3 * 2, s->cuda_stream);
                ret.data = s->converted + conv_size / 3 * 2;
        }

        return ret;
}

void nvj2k_destroy(struct nvj2k_state *s) {
        cudaFreeHost(s->bitstream_buffer);
        cudaFree(s->decode_output);
        cudaFree(s->converted);
        nvjpeg2kDecodeParamsDestroy(s->decode_params);
        nvjpeg2kStreamDestroy(s->nvjpeg2k_stream);
        nvjpeg2kDecodeStateDestroy(s->decode_state);
        nvjpeg2kDestroy(s->nvjpeg2k_handle);
        free(s);
}

static void print_j2k_info(const nvjpeg2kImageInfo_t *image_info,
                           const nvjpeg2kImageComponentInfo_t *image_comp_info)
{
        printf("Image size: %" PRIu32 "x%" PRIu32 ", tile size: %" PRIu32 "x%" PRIu32
               ", num tiles: %" PRIu32 "x%" PRIu32 ", components: %" PRIu32 "\n",
               image_info->image_width, image_info->image_height,
               image_info->tile_width, image_info->tile_height,
               image_info->num_tiles_x, image_info->num_tiles_y,
               image_info->num_components);
        for (unsigned i = 0; i < image_info->num_components; ++i) {
                printf("\tcomponent #%u size %" PRIu32 "x%" PRIu32
                       ", precision: %" PRIu8 ", sgn: %" PRIu8 "\n",
                       i, image_comp_info[i].component_width,
                       image_comp_info[i].component_height,
                       image_comp_info[i].precision, image_comp_info[i].sgn);
        }
}

static bool
validate_equal_comps(const nvjpeg2kImageInfo_t *image_info,
                     const nvjpeg2kImageComponentInfo_t *image_comp_info)
{
        // check that all comps has the same size
        for (unsigned c = 1; c < image_info->num_components; c++) {
                if (image_comp_info[c].component_width !=
                        image_comp_info[0].component_width) {
                        return false;
                }
                if (image_comp_info[c].component_height !=
                    image_comp_info[0].component_height) {
                        return false;
                }
                if (image_comp_info[c].precision !=
                    image_comp_info[0].precision) {
                        return false;
                }
                if (image_comp_info[c].sgn != image_comp_info[0].sgn) {
                        return false;
                }
        }
        return true;
}
