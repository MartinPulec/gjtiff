#include "libtiff.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>  // for cudaFree, cudaMallocManaged
#include <tiff.h>
#include <omp.h> // for omp_set_num_threads, omp_get_max_threads
#include <tiffio.h>

#include "defs.h"
#include "gdal_coords.h"
#include "libtiffinfo.hpp"
#include "kernels.h"
#include "utils.h"

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

        struct dec_image decode_fallback(TIFF *tif, struct tiff_info *tiffinfo);
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

struct dec_image libtiff_state::decode_fallback(TIFF *tif,
                                                struct tiff_info *tiffinfo)
{
        struct dec_image ret = tiffinfo->common;
        const size_t read_size = sizeof(uint32_t) * ret.width *
                                 ret.height;
        if (read_size > decoded_allocated) {
                CHECK_CUDA(cudaFreeHost(decoded));
                CHECK_CUDA(cudaFree(d_decoded));
                CHECK_CUDA(cudaMallocHost(&decoded, read_size));
                CHECK_CUDA(cudaMalloc(&d_decoded, read_size));
                decoded_allocated = read_size;
        }
        TIMER_START(TIFFReadRGBAImage, LL_DEBUG);
        /// @todo
        // TIFFReadRow{Tile,Strip} would be faster
        const int rc = TIFFReadRGBAImageOriented(
            tif, ret.width, ret.height,
            (uint32_t *)decoded, ORIENTATION_TOPLEFT, 0);
        TIMER_STOP(TIFFReadRGBAImage);
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
            tiffinfo->big_endian || tiffinfo->common.comp_count > 1 ||
            tiffinfo->bits_per_sample != 32) {
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
        TIMER_START(TIFFReadEncodedStrip, LL_DEBUG);
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
        TIMER_STOP(TIFFReadEncodedStrip);
        const size_t converted_16b_size =
            (size_t)ret.width * ret.height * ret.comp_count * 2;
        const size_t converted_8b_size =
            (size_t)ret.width * ret.height * ret.comp_count;
        const size_t converted_size = converted_16b_size + converted_8b_size;
        if (converted_size > d_converted_allocated) {
                CHECK_CUDA(cudaFree(d_converted));
                CHECK_CUDA(cudaMalloc(&d_converted, converted_size));
                d_converted_allocated = converted_size;
        }
        convert_complex_int_to_uint16(
            (int16_t *)d_decoded, (uint16_t *)d_converted, decsize / 4, stream);
        ret.data = d_converted;
        ret.scale = convert_16_8_normalize_cuda(
            &ret, d_converted + converted_16b_size, stream);
        ret.data = d_converted + converted_16b_size;
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
        TIMER_START(TIFFReadEncodedTile, LL_DEBUG);
        bool error = false;
        const char *tiff_name = TIFFFileName(tif);
        omp_set_num_threads(omp_get_max_threads());
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
        TIMER_STOP(TIFFReadEncodedTile);
        const size_t converted_size =
            (size_t) ret.height * dst_linesize * ret.comp_count;
        if (converted_size > d_converted_allocated) {
                CHECK_CUDA(cudaFree(d_converted));
                CHECK_CUDA(cudaMalloc(&d_converted, converted_size));
                d_converted_allocated = converted_size;
        }
        // convert_complex_int(d_decoded, d_converted, decsize, stream);
        ret.data = d_decoded;
        ret.scale = convert_16_8_normalize_cuda(&ret, d_converted, stream);
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
                WARN_MSG("using fallback decode for %s!\n", fname);
                ret = decode_fallback(tif, &tiffinfo);
        }
        set_suggested_size_from_gdal(fname, &ret);
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
