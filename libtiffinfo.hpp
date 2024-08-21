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
        long strip_tile_size;
        uint32_t rows_per_strip;
        uint16_t photometric;
        bool single_plane;
        uint16_t minval;
        uint16_t maxval;
};

struct tiff_info get_tiff_info(TIFF *tif);
void print_tiff_info(struct tiff_info info);

#endif // !defined LIBTIFF_HPP_2FA42E68_0054_4FAA_81AC_473D033EAADB
