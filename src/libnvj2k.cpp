#include "libnvj2k.h"

#include <cassert>             // for assert
#include <cerrno>              // for errno
#include <cinttypes>           // for PRIu32, PRIu8
#include <cstdint>             // for uint8_t
#include <cstdio>              // for fseek, fclose, fopen, fread, ftell, FILE
#include <cstdlib>             // for calloc, free
#include <cstring>             // for size_t, strerror
#include <ctime>               // for timespec
#include <cuda_runtime_api.h>  // for cudaFree, cudaFreeHost, cudaMalloc
// #include <grok.h>              // for GEO data extraction
#include <nvjpeg2k.h>

#include "defs.h"              // for dec_image, log_level, rc
#include "gdal_coords.h"       // for set_coords_from_gdal
#include "kernels.h"         // for convert_planar_rgb_to_packed
#include "utils.h"             // for ERROR_MSG, TIMER_START, TIMER_STOP

enum {
        MAX_COMPONENTS = 3,
        PIPELINE_STAGES = 10,
};

#define TILED_DECODE

struct nvj2k_state {
        nvjpeg2kHandle_t nvjpeg2k_handle;
        nvjpeg2kStream_t nvjpeg2k_stream;
        nvjpeg2kDecodeParams_t decode_params;

#ifdef TILED_DECODE
        nvjpeg2kDecodeState_t decode_states[PIPELINE_STAGES];
        cudaStream_t          decode_streams[PIPELINE_STAGES];
        cudaEvent_t           pipeline_events[PIPELINE_STAGES];
#else
        nvjpeg2kDecodeState_t decode_state;
#endif

        cudaStream_t cuda_stream;

        char *bitstream_buffer;
        size_t bitstream_buffer_allocated;

        uint8_t *decode_output;
        unsigned dec_out_linesize_allocated,
            dec_out_height_allocated;
        size_t pitch_in_bytes;

        uint8_t *converted;
        size_t converted_allocated;

        uint8_t *alpha;
        size_t alpha_allocated;
};

// prototypes
static void print_j2k_info(const nvjpeg2kImageInfo_t *image_info,
                           const nvjpeg2kImageComponentInfo_t *image_comp_info,
                           nvjpeg2kColorSpace_t color_space);
static bool
validate_equal_comps(const nvjpeg2kImageInfo_t *image_info,
                     const nvjpeg2kImageComponentInfo_t *image_comp_info);

#define CHECK_NVJPEG2K(cmd, err_action)                                        \
        do {                                                                   \
                nvjpeg2kStatus_t status = (cmd);                               \
                if (status != NVJPEG2K_STATUS_SUCCESS) {                       \
                        ERROR_MSG(#cmd " failed: %d (%s)\n", (int)status,      \
                                  nvj2k_status_to_str(status));                \
                        err_action;                                            \
                }                                                              \
        } while (0)

/**
 * @todo error handling
 */
struct nvj2k_state *nvj2k_init(cudaStream_t stream) {
        struct nvj2k_state *s = (struct nvj2k_state *) calloc(1, sizeof *s);

        nvjpeg2kCreateSimple(&s->nvjpeg2k_handle);
#ifdef TILED_DECODE
        for (int p = 0; p < PIPELINE_STAGES; p++) {
                CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(s->nvjpeg2k_handle,
                                                         &s->decode_states[p]),
                               nvj2k_destroy(s);
                               return nullptr);
                CHECK_CUDA(cudaStreamCreateWithFlags(&s->decode_streams[p],
                                                     cudaStreamNonBlocking));
                CHECK_CUDA(cudaEventCreateWithFlags(&s->pipeline_events[p],
                                                    cudaEventDisableTiming |
                                                        cudaEventBlockingSync));
                CHECK_CUDA(cudaEventRecord(s->pipeline_events[p],
                                           s->decode_streams[p]));
        }
#else
        CHECK_NVJPEG2K(
            nvjpeg2kDecodeStateCreate(s->nvjpeg2k_handle, &s->decode_state),
            nvj2k_destroy(s);
            return nullptr);
#endif
        nvjpeg2kStreamCreate(&s->nvjpeg2k_stream);
        nvjpeg2kDecodeParamsCreate(&s->decode_params);
        nvjpeg2kDecodeParamsSetOutputFormat(s->decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
        s->cuda_stream = stream;
        return s;
}

/*
static void set_coords_from_j2k(const char *fname, struct dec_image *image)
{
        grk_stream_params stream_params = {.file = fname};
#if GRK_VERSION_MAJOR >= 14
        grk_decompress_parameters parameters{};
        grk_object*c = grk_decompress_init(&stream_params, &parameters);
#else
        grk_decompress_core_params parameters{};
        grk_decompress_set_default_params(&parameters);
        grk_codec *c = grk_decompress_init(&stream_params, &parameters);
#endif
        grk_header_info info;
        bool ret = grk_decompress_read_header(c, &info);
        if (!ret) {
                WARN_MSG("Reading header with grok failed!\n");
                grk_object_unref(c);
                return;
        }

        if (info.xml_data_len == 0) {
                WARN_MSG("No J2K XML data (geo coordinates)!\n");
                return;
        }

        info.xml_data[info.xml_data_len-1] = '\0';

        const char lan_tag[] = "<LATITUDE>";
        const char lon_tag[] = "<LONGITUDE>";
        char *ptr = (char *)info.xml_data;
        for (int cidx = 0; cidx < 4; cidx++) {
                char *latitude = strstr(ptr, lan_tag);
                char *longitude = strstr(ptr, lon_tag);
                if (latitude == nullptr || longitude == nullptr ) {
                        WARN_MSG("Either lon or lat not preseent!\n");
                        grk_object_unref(c);
                        return;
                }
                latitude += strlen(lan_tag);
                longitude += strlen(lon_tag);
                image->coords[cidx].latitude = atof(latitude);
                image->coords[cidx].longitude = atof(longitude);
                assert(image->coords[cidx].latitude >= -90.1);
                assert(image->coords[cidx].latitude <= 90.1);
                assert(image->coords[cidx].longitude >= -180.1);
                assert(image->coords[cidx].longitude <= 180.1);

                ptr = MAX(latitude, longitude);
        }

        if (log_level >= LL_VERBOSE) {
                printf("Got points:\n");
                for (unsigned i = 0; i < 4; ++i) {
                        printf("\t%-11s: %f, %f\n", coord_pos_name[i],
                               image->coords[i].latitude,
                               image->coords[i].longitude);
                }
        }

        image->coords_set = true;

        grk_object_unref(c);
}
*/

struct dec_image nvj2k_decode(struct nvj2k_state *s, const char *fname,
                              bool decode_16b)
{
        FILE *f = fopen(fname, "rb");
        if (f == nullptr) {
                ERROR_MSG("Unable to open file %s: %s\n", fname, strerror(errno));
                return {};
        }
        fseek(f, 0, SEEK_END);
        const long filesize = ftell(f);
        if (s->bitstream_buffer_allocated < (size_t) filesize) {
                CHECK_CUDA(cudaFreeHost(s->bitstream_buffer));
                CHECK_CUDA(
                    cudaMallocHost((void **)&s->bitstream_buffer, filesize));
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

        nvjpeg2kColorSpace_t color_space = NVJPEG2K_COLORSPACE_UNKNOWN;
        nvjpeg2kStreamGetColorSpace(s->nvjpeg2k_stream, &color_space);

        if (!validate_equal_comps(&image_info, image_comp_info)) {
                ERROR_MSG("%s: inequal components:\n", fname);
                print_j2k_info(&image_info, image_comp_info, color_space);
                return {};
        }
        if (image_comp_info[0].sgn != 0) {
                ERROR_MSG("%s: signed samples not yet supported!\n", fname);
                print_j2k_info(&image_info, image_comp_info, color_space);
        }
        if (log_level >= LL_DEBUG) {
                print_j2k_info(&image_info, image_comp_info, color_space);
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
                CHECK_CUDA(cudaFree(s->decode_output));
                CHECK_CUDA(cudaMallocPitch((void **)&s->decode_output,
                                           &s->pitch_in_bytes,
                                           s->dec_out_linesize_allocated,
                                           s->dec_out_height_allocated));
        }

        output_image.pixel_data = (void **) &s->decode_output;
        output_image.pitch_in_bytes = &s->pitch_in_bytes;

        GPU_TIMER_START(nvjpeg2kDecode, LL_DEBUG, s->cuda_stream);
#if defined TILED_DECODE
        int buffer_index = 0;
        uint32_t tile_id = 0;
        for (uint32_t tile_y0 = 0; tile_y0 < image_info.image_height;
             tile_y0 += image_info.tile_height) {
                for (uint32_t tile_x0 = 0; tile_x0 < image_info.image_width;
                     tile_x0 += image_info.tile_width) {
                        // make sure that the previous stage are done before
                        // reusing
                        CHECK_CUDA(cudaEventSynchronize(
                            s->pipeline_events[buffer_index]));

                        nvjpeg2kImage_t nvjpeg2k_out = output_image;
                        void *pixel_data =
                            (uint8_t *)output_image.pixel_data[0] +
                            ((ptrdiff_t)tile_y0 * s->pitch_in_bytes) +
                            ((ptrdiff_t)tile_x0 * bps *
                             output_image.num_components);
                        nvjpeg2k_out.pixel_data = &pixel_data;

                        CHECK_NVJPEG2K(nvjpeg2kDecodeTile(
                                           s->nvjpeg2k_handle,
                                           s->decode_states[buffer_index],
                                           s->nvjpeg2k_stream, s->decode_params,
                                           tile_id, 0, &nvjpeg2k_out,
                                           s->decode_streams[buffer_index]),
                                       return {});

                        CHECK_CUDA(
                            cudaEventRecord(s->pipeline_events[buffer_index],
                                            s->decode_streams[buffer_index]));

                        buffer_index = (buffer_index + 1) % PIPELINE_STAGES;
                        tile_id++;
                }
        }
        for (int p = 0; p < PIPELINE_STAGES; p++) {
                CHECK_CUDA(cudaEventSynchronize(s->pipeline_events[p]));
        }
#else
        status = nvjpeg2kDecodeImage(s->nvjpeg2k_handle, s->decode_state,
                                     s->nvjpeg2k_stream, s->decode_params,
                                     &output_image, s->cuda_stream);
#endif
        GPU_TIMER_STOP(nvjpeg2kDecode);
        if (status != NVJPEG2K_STATUS_SUCCESS) {
                ERROR_MSG("Unable to decode J2K file %s: %d (%s)\n", fname,
                          (int)status, nvj2k_status_to_str(status));
                return {};
        }

        size_t conv_size = (size_t)linesize *
                           image_comp_info[0].component_height;
        conv_size += conv_size / bps;
        if (s->converted_allocated < conv_size) {
                CHECK_CUDA(cudaFree(s->converted));
                CHECK_CUDA(cudaMalloc((void **)&s->converted, conv_size * 2));
                s->converted_allocated = conv_size;
        }

        struct dec_image ret{};
        ret.width = (int)image_comp_info[0].component_width;
        ret.height = (int)image_comp_info[0].component_height;
        ret.comp_count = (int) image_info.num_components;
        ret.data = s->converted;
        ret.is_16b = bps == 2;
        set_coords_from_gdal(fname, &ret);
        const size_t sample_count = (size_t) ret.width * ret.height;

        if (s->alpha_allocated < sample_count) {
                CHECK_CUDA(cudaFree(s->alpha));
                CHECK_CUDA(cudaMalloc((void **)&s->alpha, sample_count));
                s->alpha_allocated = sample_count;
        }
        ret.alpha = s->alpha;

        if (bps == 1) { // just TCI ?
                assert(!decode_16b);
                convert_remove_pitch(
                    s->decode_output, s->converted,
                    (int)(image_comp_info[0].component_width * image_info.num_components),
                    (int)s->pitch_in_bytes, (int)image_comp_info[0].component_height,
                    s->cuda_stream);
                CHECK_CUDA(cudaMemsetAsync(ret.alpha, sample_count, 255,
                                           s->cuda_stream));
                if (ret.comp_count == 3) {
                        return ret;
                }
                // this may not happen becuase 8-bit seem to curtrently be
                // only the S2 TCI band
                normalize_8(&ret, s->converted + conv_size / 2, s->cuda_stream);
                ret.data = s->converted + conv_size / 2;
                return ret;
        }

        const uint32_t out_line_width = image_comp_info[0].component_width *
                                        image_info.num_components;
        convert_remove_pitch_16(
            (uint16_t *)s->decode_output, (uint16_t *)s->converted,
            (int)out_line_width, (int)s->pitch_in_bytes,
            (int)image_comp_info[0].component_height, s->cuda_stream);
        thrust_process_s2_band((uint16_t *)s->converted, ret.alpha,
                               sample_count, s->cuda_stream);
        if (!decode_16b) {
                ret.data = s->converted + conv_size / 3 * 2;
                thrust_16b_to_8b((uint16_t *)s->converted, ret.data,
                                 sample_count, s->cuda_stream);
                ret.is_16b = false;
        }
        // write_raw_gpu_image(s->converted, ret.width, ret.height, bps);

        return ret;
}

void nvj2k_destroy(struct nvj2k_state *s) {
        CHECK_CUDA(cudaFreeHost(s->bitstream_buffer));
        CHECK_CUDA(cudaFree(s->alpha));
        CHECK_CUDA(cudaFree(s->decode_output));
        CHECK_CUDA(cudaFree(s->converted));
        CHECK_NVJPEG2K(nvjpeg2kDecodeParamsDestroy(s->decode_params), );
        CHECK_NVJPEG2K(nvjpeg2kStreamDestroy(s->nvjpeg2k_stream), );
#ifdef TILED_DECODE
        for(int p = 0; p < PIPELINE_STAGES; p++) {
                CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(s->decode_states[p]), );
                CHECK_CUDA(cudaStreamDestroy(s->decode_streams[p]));
                CHECK_CUDA(cudaEventDestroy(s->pipeline_events[p]));
        }
#else
        CHECK_NVJPEG2K(nvjpeg2kDecodeStateDestroy(s->decode_state), );
#endif
        CHECK_NVJPEG2K(nvjpeg2kDestroy(s->nvjpeg2k_handle), );
        free(s);
}

static void print_j2k_info(const nvjpeg2kImageInfo_t *image_info,
                           const nvjpeg2kImageComponentInfo_t *image_comp_info,
                           nvjpeg2kColorSpace_t color_space)
{
        const char *cs_name = "ERROR";
        switch (color_space) {
        case NVJPEG2K_COLORSPACE_NOT_SUPPORTED: cs_name = "unsupp"; break;
        case NVJPEG2K_COLORSPACE_UNKNOWN:       cs_name = "unknown"; break;
        case NVJPEG2K_COLORSPACE_SRGB:          cs_name = "sRGB"; break;
        case NVJPEG2K_COLORSPACE_GRAY:          cs_name = "gray"; break;
        case NVJPEG2K_COLORSPACE_SYCC:          cs_name = "SYCC"; break;
        }

        INFO_MSG("Image size: %" PRIu32 "x%" PRIu32 ", tile size: %" PRIu32
                 "x%" PRIu32 ", num tiles: %" PRIu32 "x%" PRIu32
                 ", components: %" PRIu32 ", color space: %s\n",
                 image_info->image_width, image_info->image_height,
                 image_info->tile_width, image_info->tile_height,
                 image_info->num_tiles_x, image_info->num_tiles_y,
                 image_info->num_components, cs_name);
        for (unsigned i = 0; i < image_info->num_components; ++i) {
                INFO_MSG("\tcomponent #%u size %" PRIu32 "x%" PRIu32
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
