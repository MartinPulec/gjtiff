#include "libtiff.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>  // for cudaFree, cudaMallocManaged
#include <tiff.h>
#include <tiffio.h>

#include "defs.h"
#include "libtiffinfo.hpp"
#include "kernels.hpp"
#include "utils.hpp"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

void nullTIFFErrorHandler(const char *, const char *, va_list) {}

struct libtiff_state {
        libtiff_state(int l, cudaStream_t stream);
        ~libtiff_state();

        int log_level;
        cudaStream_t stream;

        uint8_t *decoded = nullptr;
        uint8_t *d_decoded = nullptr;
        size_t decoded_allocated{};

        uint8_t *d_converted = nullptr;
        size_t d_converted_allocated = 0;

        struct dec_image decode(const char *fname);

        struct dec_image decode_fallback(TIFF *tif);
        struct dec_image decode_stripped_complex(TIFF *tif, struct tiff_info *tiffinfo);
        struct dec_image decode_tiled(TIFF *tif, struct tiff_info *tiffinfo);
};

libtiff_state::libtiff_state(int l, cudaStream_t s) : log_level(l), stream(s)
{
        if (l == 0) {
                TIFFSetWarningHandler(nullTIFFErrorHandler);
        }
}

libtiff_state::~libtiff_state()
{
        CHECK_CUDA(cudaFreeHost(decoded));
        CHECK_CUDA(cudaFree(d_decoded));
        CHECK_CUDA(cudaFree(d_converted));
}

struct dec_image libtiff_state::decode_fallback(TIFF *tif)
{
        struct tiff_info tiffinfo = get_tiff_info(tif);
        struct dec_image ret = tiffinfo.common;
        const size_t read_size = sizeof(uint32_t) * ret.width *
                                 ret.height;
        if (read_size > decoded_allocated) {
                CHECK_CUDA(cudaFreeHost(decoded));
                CHECK_CUDA(cudaFree(d_decoded));
                CHECK_CUDA(cudaMallocHost(&decoded, read_size));
                CHECK_CUDA(cudaMalloc(&d_decoded, read_size));
                decoded_allocated = read_size;
        }
        TIMER_START(TIFFReadRGBAImage, log_level);
        /// @todo
        // TIFFReadRow{Tile,Strip} would be faster
        const int rc = TIFFReadRGBAImageOriented(
            tif, ret.width, ret.height,
            (uint32_t *)decoded, ORIENTATION_TOPLEFT, 0);
        TIMER_STOP(TIFFReadRGBAImage, log_level);
        if (rc != 1) {
                ERROR_MSG("libtiff decode image %s failed!\n",
                          TIFFFileName(tif));
                return {};
        }
        CHECK_CUDA(cudaMemcpyAsync(d_decoded, decoded, read_size,
                                   cudaMemcpyHostToDevice,
                                   (cudaStream_t)stream));
        const size_t out_size = (size_t)ret.width * ret.height * ret.comp_count;
        if (out_size > d_converted_allocated) {
                CHECK_CUDA(cudaFree(d_converted));
                CHECK_CUDA(cudaMalloc(&d_converted, out_size));
                d_converted_allocated = out_size;
        }
        switch (ret.comp_count) {
        case 1:
                convert_rgba_grayscale(d_decoded, d_converted,
                                       (size_t)ret.width * ret.height, stream);
                break;
        case 3:
                convert_rgba_rgb(d_decoded, d_converted,
                                 (size_t)ret.width * ret.height, stream);
                break;
        default:
                ERROR_MSG("Unsupported sample count %d!\n", ret.comp_count);
                return {};
        }
        ret.data = d_converted;
        return ret;
}

struct dec_image
libtiff_state::decode_stripped_complex(TIFF *tif, struct tiff_info *tiffinfo)
{
        if (tiffinfo->sample_format != SAMPLEFORMAT_COMPLEXINT ||
            tiffinfo->big_endian) {
                return {};
        }

        struct dec_image ret = tiffinfo->common;

        assert(tiffinfo->strip_tile_size > 0);
        const tstrip_t numberOfStrips = TIFFNumberOfStrips(tif);
        size_t decsize = tiffinfo->strip_tile_size * numberOfStrips;
        if (decsize > decoded_allocated) {
                CHECK_CUDA(cudaFreeHost(decoded));
                CHECK_CUDA(cudaFree(d_decoded));
                CHECK_CUDA(cudaMallocHost(&decoded, decsize));
                CHECK_CUDA(cudaMalloc(&d_decoded, decsize));
                decoded_allocated = decsize;
        }
        TIMER_START(TIFFReadEncodedStrip, log_level);
        for (tstrip_t strip = 0; strip < numberOfStrips; ++strip) {
                size_t offset = strip * tiffinfo->strip_tile_size;
                const tmsize_t rc = TIFFReadEncodedStrip(
                    tif, strip, decoded + offset, tiffinfo->strip_tile_size);
                if (rc == -1) {
                        ERROR_MSG("libtiff decode image %s failed!\n",
                                  TIFFFileName(tif));
                        return {};
                }
                CHECK_CUDA(cudaMemcpyAsync(d_decoded + offset, decoded + offset,
                                           tiffinfo->strip_tile_size,
                                           cudaMemcpyHostToDevice, stream));
        }
        TIMER_STOP(TIFFReadEncodedStrip, log_level);
        const size_t converted_size = (size_t)ret.width * ret.height * ret.comp_count;
        if (converted_size > d_converted_allocated) {
                CHECK_CUDA(cudaFree(d_converted));
                CHECK_CUDA(cudaMalloc(&d_converted, converted_size));
                d_converted_allocated = converted_size;
        }
        convert_complex_int(d_decoded, d_converted, decsize, stream);
        ret.data = d_converted;
        return ret;
}

struct dec_image
libtiff_state::decode_tiled(TIFF *tif, struct tiff_info *tiffinfo)
{
        if (tiffinfo->sample_format != SAMPLEFORMAT_UINT ||
            tiffinfo->big_endian || tiffinfo->common.comp_count > 1 ||
            tiffinfo->bits_per_sample != 16) {
                return {};
        }

        struct dec_image ret = tiffinfo->common;

        assert(tiffinfo->strip_tile_size > 0);
        const tstrip_t numberOfTiles = TIFFNumberOfTiles(tif);
        size_t decsize = tiffinfo->strip_tile_size * numberOfTiles;
        if (decsize > decoded_allocated) {
                CHECK_CUDA(cudaFreeHost(decoded));
                CHECK_CUDA(cudaFree(d_decoded));
                CHECK_CUDA(cudaMallocHost(&decoded, decsize));
                CHECK_CUDA(cudaMalloc(&d_decoded, decsize));
                decoded_allocated = decsize;
        }

        const int tile_count_vertical =
            (tiffinfo->common.width + tiffinfo->tile_width - 1) /
            tiffinfo->tile_width;
        const int bps = tiffinfo->bits_per_sample / 8;
        const int dst_linesize = tiffinfo->common.width * bps;
        const int tile_linesize = tiffinfo->tile_width * bps;
        TIMER_START(TIFFReadEncodedTile, log_level);
        bool error = false;
        const char *tiff_name = TIFFFileName(tif);
        #pragma omp parallel for
        for (tstrip_t tile = 0; tile < numberOfTiles; ++tile) {
                size_t offset = tile * tiffinfo->strip_tile_size;
                uint8_t *tile_data = decoded + offset;
                TIFF *ttif = TIFFOpen(tiff_name, "r");
                const tmsize_t rc = TIFFReadEncodedTile(
                    ttif, tile, tile_data, tiffinfo->strip_tile_size);
                if (rc == -1) {
                        continue;
                }
                size_t offset_x =
                    (tile % tile_count_vertical) * tiffinfo->tile_width;
                size_t offset_y =
                    (tile / tile_count_vertical) * tiffinfo->tile_height;
                size_t dst_offset = offset_y * dst_linesize + offset_x * bps;
                const int rows =
                    MIN(tiffinfo->tile_height, ret.height - offset_y);
                const int columns =
                    MIN(tiffinfo->tile_width, ret.width - offset_x);
                CHECK_CUDA(cudaMemcpy2DAsync(
                    d_decoded + dst_offset, dst_linesize, decoded + offset,
                    tile_linesize, (size_t)columns * bps, rows,
                    cudaMemcpyHostToDevice, stream));
                TIFFClose(ttif);
        }
        if (error) {
                ERROR_MSG("libtiff decode image %s failed!\n",
                          TIFFFileName(tif));
                return {};
        }
        TIMER_STOP(TIFFReadEncodedTile, log_level);
        const size_t converted_size =
            (size_t) ret.height * dst_linesize * ret.comp_count;
        if (converted_size > d_converted_allocated) {
                CHECK_CUDA(cudaFree(d_converted));
                CHECK_CUDA(cudaMalloc(&d_converted, converted_size));
                d_converted_allocated = converted_size;
        }
        // convert_complex_int(d_decoded, d_converted, decsize, stream);
        ret.data = d_decoded;
        convert_16_8_cuda(&ret, d_converted, stream);
        ret.data = d_converted;
        return ret;
}

struct dec_image libtiff_state::decode(const char *fname)
{
        TIFF *tif = TIFFOpen(fname, "r");
        if (tif == nullptr) {
                ERROR_MSG("libtiff cannot open image %s!\n", fname);
                return {};
        }
        struct tiff_info tiffinfo = get_tiff_info(tif);
        // image_info->bits_per_sample[0] = 8;
        if (log_level > 1) {
                print_tiff_info(tiffinfo);
        }

        struct dec_image ret = tiffinfo.tiled
                                   ? decode_tiled(tif, &tiffinfo)
                                   : decode_stripped_complex(tif, &tiffinfo);

        if (ret.data == nullptr) {
                ret = decode_fallback(tif);
        }
        TIFFClose(tif);

        return ret;
}

struct libtiff_state *libtiff_create(int log_level, void *stream) {
        return new libtiff_state(log_level, (cudaStream_t) stream);
}

struct dec_image libtiff_decode(struct libtiff_state *s, const char *fname)
{
        return s->decode(fname);
}

void libtiff_destroy(struct libtiff_state *s) {
        delete s;
}
