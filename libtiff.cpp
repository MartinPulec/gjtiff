#include "libtiff.hpp"

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>  // for cudaFree, cudaMallocManaged
#include <tiff.h>
#include <tiffio.h>

#include "defs.h"
#include "libtiffinfo.hpp"
#include "kernels.hpp"
#include "utils.hpp"

using std::unique_ptr;

void nullTIFFErrorHandler(const char *, const char *, va_list) {}

libtiff_state::libtiff_state(int l) : log_level(l)
{
        if (l == 0) {
                TIFFSetWarningHandler(nullTIFFErrorHandler);
        }
}

struct dec_image libtiff_state::decode(const char *fname, void *stream)
{
        TIFF *tif = TIFFOpen(fname, "r");
        if (tif == nullptr) {
                fprintf(stderr, "libtiff cannot open image %s!\n", fname);
                return {};
        }
        struct tiff_info tiffinfo = get_tiff_info(tif);
        struct dec_image ret = tiffinfo.common;
        // image_info->bits_per_sample[0] = 8;
        if (log_level > 1) {
                print_tiff_info(tiffinfo);
        }
        const size_t read_size = sizeof(uint32_t) * ret.width *
                                 ret.height;
        if (read_size > tmp_buffer_allocated) {
                void *ptr = nullptr;
                cudaMallocManaged(&ptr, read_size);
                auto cfree = [](void *ptr) { cudaFree(ptr); };
                tmp_buffer = unique_ptr<uint8_t[], void (*)(void *)>(
                    (uint8_t *)ptr, cfree);
                assert(tmp_buffer.get() != nullptr);
                tmp_buffer_allocated = read_size;
        }
        TIMER_START(TIFFReadRGBAImage, log_level);
        /// @todo
        // TIFFReadRow{Tile,Strip} would be faster
        const int rc = TIFFReadRGBAImageOriented(
            tif, ret.width, ret.height,
            (uint32_t *)tmp_buffer.get(), ORIENTATION_TOPLEFT, 0);
        TIMER_STOP(TIFFReadRGBAImage, log_level);
        TIFFClose(tif);
        if (rc != 1) {
                fprintf(stderr, "libtiff decode image %s failed!\n", fname);
                return {};
        }
        const size_t out_size = (size_t)ret.width * ret.height * ret.comp_count;
        if (out_size > decoded_allocated) {
                void *ptr = nullptr;
                cudaMallocManaged(&ptr, read_size);
                assert(ptr != nullptr);
                auto cfree = [](void *ptr) { cudaFree(ptr); };
                decoded = unique_ptr<uint8_t[], void (*)(void *)>(
                    (uint8_t *)ptr, cfree);
                decoded_allocated = out_size;
        }
        switch (ret.comp_count) {
        case 1:
                convert_rgba_grayscale(tmp_buffer.get(), decoded.get(),
                                       (size_t)ret.width * ret.height, stream);
                break;
        case 3:
                convert_rgba_rgb(tmp_buffer.get(), decoded.get(),
                                 (size_t)ret.width * ret.height, stream);
                break;
        default:
                fprintf(stderr, "Unsupported sample count %d!\n",
                        ret.comp_count);
                return {};
        }
        ret.rc = SUCCESS;
        ret.data = decoded.get();
        return ret;
}
