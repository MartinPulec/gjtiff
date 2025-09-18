#include "libtiffinfo.hpp"

#include <cstring>
#include <tiffio.h>

#include "defs.h"

struct val_str_desc_map {
        int val;
        const char *str;
        const char *desc;
};
#define MACRO_TO_STR(x) x,#x
struct val_str_desc_map photometric[] = {
    {MACRO_TO_STR(PHOTOMETRIC_MINISWHITE), "min value is white"},
    {MACRO_TO_STR(PHOTOMETRIC_MINISBLACK), "min value is black"},
    {MACRO_TO_STR(PHOTOMETRIC_RGB), "RGB color model"},
    {MACRO_TO_STR(PHOTOMETRIC_PALETTE), "color map indexed"},
    {MACRO_TO_STR(PHOTOMETRIC_MASK), "$holdout mask"},
    {MACRO_TO_STR(PHOTOMETRIC_SEPARATED), "!color separations"},
    {MACRO_TO_STR(PHOTOMETRIC_YCBCR), "!CCIR 601"},
    {MACRO_TO_STR(PHOTOMETRIC_CIELAB), "!1976 CIE L*a*b*"},
    {MACRO_TO_STR(PHOTOMETRIC_ICCLAB), "ICC L*a*b* [Adobe TIFF Technote 4]"},
    {MACRO_TO_STR(PHOTOMETRIC_ITULAB), "ITU L*a*b*"},
    {MACRO_TO_STR(PHOTOMETRIC_CFA), "color filter array"},
    {MACRO_TO_STR(PHOTOMETRIC_LOGL), "CIE Log2(L)"},
    {MACRO_TO_STR(PHOTOMETRIC_LOGLUV), "CIE Log2(L) (u',v')"},
};
struct val_str_desc_map sample_fmts[] = {
    {MACRO_TO_STR(SAMPLEFORMAT_UINT), "!unsigned integer data"},
    {MACRO_TO_STR(SAMPLEFORMAT_INT), "!signed integer data"},
    {MACRO_TO_STR(SAMPLEFORMAT_IEEEFP), "!IEEE floating point data"},
    {MACRO_TO_STR(SAMPLEFORMAT_VOID), "!untyped data"},
    {MACRO_TO_STR(SAMPLEFORMAT_COMPLEXINT), "!complex signed int"},
    {MACRO_TO_STR(SAMPLEFORMAT_COMPLEXIEEEFP), "!complex ieee floating"},
};
struct val_str_desc_map compressions[] = {
    {MACRO_TO_STR(COMPRESSION_NONE), "dump mode"},
    {MACRO_TO_STR(COMPRESSION_CCITTRLE), "CCITT modified Huffman RLE"},
    {MACRO_TO_STR(COMPRESSION_CCITTFAX3), "CCITT Group 3 fax encoding"},
    {MACRO_TO_STR(COMPRESSION_CCITT_T4), "CCITT T.4 (TIFF 6 name)"},
    {MACRO_TO_STR(COMPRESSION_CCITTFAX4), "CCITT Group 4 fax encoding"},
    {MACRO_TO_STR(COMPRESSION_CCITT_T6), "CCITT T.6 (TIFF 6 name)"},
    {MACRO_TO_STR(COMPRESSION_LZW), "Lempel-Ziv  & Welch"},
    {MACRO_TO_STR(COMPRESSION_OJPEG), "!6.0 JPEG"},
    {MACRO_TO_STR(COMPRESSION_JPEG), "%JPEG DCT compression"},
    {MACRO_TO_STR(COMPRESSION_T85), "!TIFF/FX T.85 JBIG compression"},
    {MACRO_TO_STR(COMPRESSION_T43),
     "!TIFF/FX T.43 colour by layered JBIG compression"},
    {MACRO_TO_STR(COMPRESSION_NEXT), "NeXT 2-bit RLE"},
    {MACRO_TO_STR(COMPRESSION_CCITTRLEW), "#1 w/ word alignment"},
    {MACRO_TO_STR(COMPRESSION_PACKBITS), "Macintosh RLE"},
    {MACRO_TO_STR(COMPRESSION_THUNDERSCAN), "ThunderScan RLE"},
    {MACRO_TO_STR(COMPRESSION_IT8CTPAD), "IT8 CT w/padding"},
    {MACRO_TO_STR(COMPRESSION_IT8LW), "IT8 Linework RLE"},
    {MACRO_TO_STR(COMPRESSION_IT8MP), "IT8 Monochrome picture"},
    {MACRO_TO_STR(COMPRESSION_IT8BL), "IT8 Binary line art"},
    {MACRO_TO_STR(COMPRESSION_PIXARFILM), "Pixar companded 10bit LZW"},
    {MACRO_TO_STR(COMPRESSION_PIXARLOG), "Pixar companded 11bit ZIP"},
    {MACRO_TO_STR(COMPRESSION_DEFLATE), "Deflate compression, legacy tag"},
    {MACRO_TO_STR(COMPRESSION_ADOBE_DEFLATE),
     "Deflate compression, as recognized by Adobe"},
    {MACRO_TO_STR(COMPRESSION_DCS), "Kodak DCS encoding"},
    {MACRO_TO_STR(COMPRESSION_JBIG), "ISO JBIG"},
    {MACRO_TO_STR(COMPRESSION_SGILOG), "SGI Log Luminance RLE"},
    {MACRO_TO_STR(COMPRESSION_SGILOG24), "SGI Log 24-bit packed"},
    {MACRO_TO_STR(COMPRESSION_JP2000), "Leadtools JPEG2000"},
    {MACRO_TO_STR(COMPRESSION_LERC),
     "ESRI Lerc codec: https://github.com/Esri/lerc"},
    {MACRO_TO_STR(COMPRESSION_LZMA), "LZMA2"},
    {MACRO_TO_STR(COMPRESSION_ZSTD),
     "ZSTD: WARNING not registered in Adobe-maintained registry"},
    {MACRO_TO_STR(COMPRESSION_WEBP),
     "WEBP: WARNING not registered in Adobe-maintained registry"},
    {MACRO_TO_STR(COMPRESSION_JXL),
     "JPEGXL: WARNING not registered in Adobe-maintained registry"},
};

bool tiff_get_corners(const double *points, size_t count, int img_width,
                      int img_height, struct coordinate coords[4])
{
        unsigned points_set = 0;
        for (unsigned i = 0; i < count; i += 6) {
                double x = points[i];
                double y = points[i + 1];

                if (x == 0 || x == img_width - 1) {
                        if (y == 0 || y == img_height - 1) {
                                int idx = -1;
                                if (y == 0) {
                                        idx = x == 0 ? 0 : 1;
                                } else {
                                        idx = x == 0 ? 3 : 2;
                                }

                                points_set |= 1 << idx;
                                coords[idx].latitude = points[i + 4];
                                coords[idx].longitude = points[i + 3];

                                if (points_set == 0xF) { // all points set
                                        break;
                                }
                        }
                }
        }

        if (points_set != 0xF) {
                WARN_MSG("Not all coordinate points were set!\n");
                return false;
        }

        if (log_level >= LL_VERBOSE) {
                VERBOSE_MSG("Got points:\n");
                for (unsigned i = 0; i < 4; ++i) {
                        VERBOSE_MSG("\t%-11s: %f, %f\n", coord_pos_name[i],
                               coords[i].latitude,
                               coords[i].longitude);
                }
        }
        return true;
}

struct tiff_info get_tiff_info(TIFF *tif)
{
        struct tiff_info ret {
        };
        uint32_t uval;
        uint16_t u16val;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &uval);
        ret.common.width = uval;
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &uval);
        ret.common.height = uval;
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &uval);
        ret.common.comp_count = uval;
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &ret.bits_per_sample);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &ret.sample_format);
        TIFFGetField(tif, TIFFTAG_COMPRESSION, &ret.compression);
        ret.tiled = TIFFIsTiled(tif);
        if (ret.tiled) {
                TIFFGetField(tif, TIFFTAG_TILEWIDTH, &ret.tile_width);
                TIFFGetField(tif, TIFFTAG_TILELENGTH, &ret.tile_height);
        } else {
                TIFFGetField(tif, TIFFTAG_ROWSPERSTRIP, &ret.rows_per_strip);
        }
        ret.strip_tile_size =  ret.tiled ? TIFFTileSize(tif) : TIFFStripSize(tif);
        TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &ret.photometric);
        TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &u16val);
        ret.single_plane = u16val == PLANARCONFIG_CONTIG;
        ret.big_endian = TIFFIsBigEndian(tif) != 0;
        TIFFGetField(tif, TIFFTAG_MAXSAMPLEVALUE, &ret.maxval);
        TIFFGetField(tif, TIFFTAG_MINSAMPLEVALUE, &ret.minval);

        const char *description = nullptr;
        if (TIFFGetField(tif, TIFFTAG_IMAGEDESCRIPTION, &description) == 1 &&
            strstr(description, " SLC ") != nullptr) {
                ret.common.is_slc = true;
        }

        double *tiepoints = nullptr;
        uint32_t count = 0;
        if (TIFFGetField(tif, TIFFTAG_MODELTIEPOINTTAG, &count, &tiepoints) == 1) {
                ret.common.tie_point_count = count;
                ret.common.tie_points = tiepoints;
                if (tiff_get_corners(tiepoints, count, ret.common.width,
                                     ret.common.height, ret.common.coords)) {
                        ret.common.coords_set = true;
                }
        };

        return ret;
}

void print_tiff_info(struct tiff_info info)
{
        INFO_MSG("width: %u\n", info.common.width);
        INFO_MSG("height: %u\n", info.common.height);
        INFO_MSG("components: %u\n", info.common.comp_count);
        INFO_MSG("bits_per_sample: %hu\n", info.bits_per_sample);
        for (unsigned i = 0; i < ARR_SIZE(sample_fmts); ++i) {
                if (sample_fmts[i].val == info.sample_format) {
                        INFO_MSG("sample format: %s (%s)\n", sample_fmts[i].str,
                               sample_fmts[i].desc);
                        break;
                }
        }
        for (unsigned i = 0; i < ARR_SIZE(compressions); ++i) {
                if (compressions[i].val == info.compression) {
                        INFO_MSG("compression: %s (%s)\n", compressions[i].str,
                               compressions[i].desc);
                        break;
                }
        }
        INFO_MSG("range: %hu-%hu\n", info.minval, info.maxval);
        INFO_MSG("tiled: %d\n", (int) info.tiled);
        if (info.tiled) {
                INFO_MSG("tile width/height: %ux%u\n", info.tile_width,
                       info.tile_height);
        } else {
                INFO_MSG("rows per strip: %u\n", info.rows_per_strip);
        }
        INFO_MSG("strip or tile size: %ld B\n", info.strip_tile_size);
        for (unsigned i = 0; i < ARR_SIZE(photometric); ++i) {
                if (photometric[i].val == info.photometric) {
                        INFO_MSG("photometric: %s (%s)\n", photometric[i].str,
                               photometric[i].desc);
                        break;
                }
        }
        INFO_MSG("single plane: %d\n", (int) info.single_plane);
        INFO_MSG("big endian: %d\n", (int) info.big_endian);
}

