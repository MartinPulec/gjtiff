#ifndef LIBTIFF_HPP_2FA42E68_0054_4FAA_81AC_473D033EAADB
#define LIBTIFF_HPP_2FA42E68_0054_4FAA_81AC_473D033EAADB

#include <cstdint>
#include <tiffio.h>

#include "defs.h"

struct tiff_info {
        struct dec_image common;
        uint16_t bits_per_sample;
        uint16_t sample_format;
        uint16_t compression;
        bool tiled;
        tmsize_t strip_tile_size;
        union {
                struct {
                        uint32_t rows_per_strip;
                };
                struct {
                        uint32_t tile_width;
                        uint32_t tile_height;
                };
        };
        uint16_t photometric;
        bool single_plane;
        bool big_endian;
        bool is_slc;
        uint16_t minval;
        uint16_t maxval;
};

struct tiff_info get_tiff_info(TIFF *tif);
void print_tiff_info(struct tiff_info info);
bool tiff_get_corners_bounds(const double *points, size_t count, int img_width,
                             int img_height, struct coordinate coords[4],
                             double bounds[4]);

#endif // !defined LIBTIFF_HPP_2FA42E68_0054_4FAA_81AC_473D033EAADB
