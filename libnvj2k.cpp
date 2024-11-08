#include "libnvj2k.h"

#include <cerrno>              // for errno
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
        NUM_COMPONENTS = 3,
};

struct nvj2k_state {
        nvjpeg2kHandle_t nvjpeg2k_handle;
        nvjpeg2kStream_t nvjpeg2k_stream;
        nvjpeg2kDecodeState_t decode_state;

        cudaStream_t cuda_stream;

        char *bitstream_buffer;
        size_t bitstream_buffer_allocated;

        uint8_t *decode_output[NUM_COMPONENTS];
        unsigned decode_output_width,
            decode_output_height;
        size_t pitch_in_bytes[NUM_COMPONENTS];

        uint8_t *converted;
        size_t converted_allocated;
};

struct nvj2k_state *nvj2k_init(cudaStream_t stream) {
        struct nvj2k_state *s = (struct nvj2k_state *) calloc(1, sizeof *s);

        nvjpeg2kCreateSimple(&s->nvjpeg2k_handle);
        nvjpeg2kDecodeStateCreate(s->nvjpeg2k_handle, &s->decode_state);
        nvjpeg2kStreamCreate(&s->nvjpeg2k_stream);
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

        if (image_info.num_components != NUM_COMPONENTS) {
                ERROR_MSG("%s: assumed %d components, got %d!\n", fname,
                          (int)NUM_COMPONENTS, image_info.num_components);
                return {};
        }

        // assuming the decoding of images with 8 bit precision, and 3
        // components

        nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];

        for (unsigned c = 0; c < image_info.num_components; c++) {
                nvjpeg2kStreamGetImageComponentInfo(s->nvjpeg2k_stream,
                                                    &image_comp_info[c], c);
        }

        nvjpeg2kImage_t output_image;

        if (s->decode_output_width != image_comp_info[0].component_width ||
            s->decode_output_height != image_comp_info[0].component_height) {
                for (int i = 0; i < NUM_COMPONENTS; i++) {
                        cudaFree(s->decode_output[i]);
                        cudaMallocPitch((void **)&s->decode_output[i],
                                        &s->pitch_in_bytes[i],
                                        image_comp_info[0].component_width,
                                        image_comp_info[0].component_height);
                }
        }
        // check that all comps has the same size
        for (int c = 1; c < NUM_COMPONENTS; c++) {
                if (image_comp_info[c].component_width ==
                        image_comp_info[0].component_width &&
                    image_comp_info[c].component_height ==
                        image_comp_info[0].component_height) {
                        continue;
                }
                ERROR_MSG("%s: component #%d size %ux%u doesn't equal "
                          "the first component (%ux%u)!\n",
                          fname, c, image_comp_info[c].component_width,
                          image_comp_info[c].component_height,
                          image_comp_info[0].component_width,
                          image_comp_info[0].component_height);
                return {};
        }
        s->decode_output_width = image_comp_info[0].component_width;
        s->decode_output_height = image_comp_info[0].component_height;

        output_image.pixel_data = (void **) s->decode_output;
        output_image.pixel_type = NVJPEG2K_UINT8;
        output_image.pitch_in_bytes = s->pitch_in_bytes;
        output_image.num_components = image_info.num_components;

        TIMER_START(nvjpeg2kDecode, log_level);
        status =
            nvjpeg2kDecode(s->nvjpeg2k_handle, s->decode_state,
                           s->nvjpeg2k_stream, &output_image, s->cuda_stream);
        TIMER_STOP(nvjpeg2kDecode, log_level);
        if (status != NVJPEG2K_STATUS_SUCCESS) {
                ERROR_MSG("Unable to decode J2K file %s: %d\n", fname,
                          (int)status);
                return {};
        }

        size_t conv_size = (size_t)s->decode_output_width *
                          s->decode_output_height * NUM_COMPONENTS;
        if (s->converted_allocated < conv_size) {
                cudaFree(s->converted);
                cudaMalloc((void **)&s->converted, conv_size);
                s->converted_allocated = conv_size;
        }

        convert_planar_rgb_to_packed(
            s->decode_output, s->converted, s->decode_output_width,
            s->pitch_in_bytes[0], s->decode_output_height, s->cuda_stream);

        const struct dec_image ret = {(enum rc)0, (int)s->decode_output_width,
                                      (int)s->decode_output_height,
                                      NUM_COMPONENTS, s->converted};
        return ret;
}

void nvj2k_destroy(struct nvj2k_state *s) {
        cudaFreeHost(s->bitstream_buffer);
        for (int c = 0; c < NUM_COMPONENTS; c++) {
                cudaFree(s->decode_output[c]);
        }
        cudaFree(s->converted);
        nvjpeg2kStreamDestroy(s->nvjpeg2k_stream);
        nvjpeg2kDecodeStateDestroy(s->decode_state);
        nvjpeg2kDestroy(s->nvjpeg2k_handle);
        free(s);
}
