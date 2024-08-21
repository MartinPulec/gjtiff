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

void nullTIFFErrorHandler(const char *, const char *, va_list) {}

struct libtiff_state {
        libtiff_state(int l);
        ~libtiff_state();
        int log_level;

        uint8_t *decoded = nullptr;
        uint8_t *d_decoded = nullptr;
        size_t decoded_allocated{};

        uint8_t *d_converted = nullptr;
        size_t d_converted_allocated = 0;

        struct dec_image decode(const char *fname, cudaStream_t stream);

        struct dec_image decode_fallback(TIFF *tif, cudaStream_t stream);
};

libtiff_state::libtiff_state(int l) : log_level(l)
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

struct dec_image libtiff_state::decode_fallback(TIFF *tif, cudaStream_t stream)
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

struct dec_image libtiff_state::decode(const char *fname, cudaStream_t stream)
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

        struct dec_image ret = decode_fallback(tif, stream);
        TIFFClose(tif);
        return ret;
}

struct libtiff_state *libtiff_create(int log_level) {
        return new libtiff_state(log_level);
}

struct dec_image libtiff_decode(struct libtiff_state *s, const char *fname,
                                void *stream) {
        return s->decode(fname, (cudaStream_t) stream);
}

void libtiff_destroy(struct libtiff_state *s) {
        delete s;
}
