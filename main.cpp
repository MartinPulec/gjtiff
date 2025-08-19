/*
 * Copyright (c) 2024 CESNET
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>      // for std::min
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <libgpujpeg/gpujpeg_type.h>
#include <libgpujpeg/gpujpeg_version.h>
#include <linux/limits.h>
#include <omp.h>
#include <sys/stat.h> // for mkdir
#include <unistd.h>

#include "defs.h"
#include "downscaler.h"
#include "kernels.h"
#include "libnvj2k.h"
#include "libnvtiff.h"
#include "libtiff.hpp"
#include "pam.h"
#include "rotate.h"
#include "utils.h"
#include "webp.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define STRINGIFY(A) #A
#define TOSTRING(A) STRINGIFY(A)
#ifndef GIT_REV
#define GIT_REV (unknown)
#endif
#define GIT_REV_STR TOSTRING(GIT_REV)

enum { MAX_ZOOM_COUNT = 20,};

// declared in defs.h
int log_level = 0;
size_t gpu_memory = 0;
cudaEvent_t cuda_event_start;
cudaEvent_t cuda_event_stop;

int interpolation = 0;
long long mem_limit = 0;
enum out_format output_format = OUTF_JPEG;

struct options {
        int req_quality;      ///< 0-100; -1 default
        int downscale_factor;
        bool use_libtiff;
        bool norotate;
        bool no_whole_image;
        int zoom_levels[MAX_ZOOM_COUNT];
#define OPTIONS_INIT {-1, 1, false, false, false, {-1}}
};

struct state_gjtiff {
        state_gjtiff(struct options opts);
        ~state_gjtiff();
        struct options opts;
        bool use_libtiff; // if nvCOMP not found, enforce libtiff
        cudaStream_t stream;
        struct nvj2k_state *state_nvj2k;
        struct nvtiff_state *state_nvtiff;
        struct libtiff_state *state_libtiff;
        // GPUJPEG
        struct gpujpeg_encoder *gj_enc{};
        struct webp_encoder *webp_enc{};
        // downscaler
        struct downscaler_state *downscaler = NULL;
        // rotate
        struct rotate_state *rotate = NULL;

        bool first = true;
};

static const char *get_ext()
{
        switch (output_format) {
        case OUTF_JPEG:
                return ".jpg";
        case OUTF_WEBP:
                return ".webp";
        case OUTF_RAW:
                return ".pnm";
        }
        abort();
}

state_gjtiff::state_gjtiff(struct options opts)
    : opts(opts)
{
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&cuda_event_start));
        CHECK_CUDA(cudaEventCreate(&cuda_event_stop));

        state_libtiff = libtiff_create(log_level, stream);
        state_nvj2k = nvj2k_init(stream);
        assert(state_nvj2k != nullptr);
        state_nvtiff = nvtiff_init(stream, log_level);
        assert(state_nvtiff != nullptr);
        if (output_format == OUTF_JPEG) {
                gj_enc = gpujpeg_encoder_create(stream);
                assert(gj_enc != nullptr);
        }
        if (output_format == OUTF_WEBP) {
                webp_enc = webp_encoder_create(opts.req_quality);
                assert(webp_enc != nullptr);
        }
        downscaler = downscaler_init(stream);
        assert(downscaler != nullptr);
        if (!opts.norotate) {
                rotate = rotate_init(stream);
                assert(rotate != nullptr);
        }
}

state_gjtiff::~state_gjtiff()
{
        if (gj_enc != NULL) {
                gpujpeg_encoder_destroy(gj_enc);
        }
        webp_encoder_destroy(webp_enc);
        nvj2k_destroy(state_nvj2k);
        nvtiff_destroy(state_nvtiff);
        libtiff_destroy(state_libtiff);
        downscaler_destroy(downscaler);
        rotate_destroy(rotate);

        CHECK_CUDA(cudaEventDestroy(cuda_event_start));
        CHECK_CUDA(cudaEventDestroy(cuda_event_stop));
        // destroy last - components may hold the stream
        CHECK_CUDA(cudaStreamDestroy(stream));
}

/**
 * Decodes TIFF using nvTIFF with libtiff fallback
 *
 * If DEFLATE-compressed TIFF is detected but nvCOMP not found in
 * library lookup path (LD_LIBRARY_PATH on Linux), exit() is called
 * unless enforced use of libtiff.
 *
 * If nvTIFF reports unsupported file, libtiff fallback is used regardless
 * use_tiff is set.
 */
static dec_image decode_tiff(struct state_gjtiff *s, const char *fname)
{
        struct dec_image dec = nvtiff_decode(s->state_nvtiff, fname);
        if (dec.data != nullptr) {
                return dec;
        }
        if (dec.rc == ERR_NVCOMP_NOT_FOUND) {
                if (!s->use_libtiff) {
                        ERROR_MSG(
                            "Use option '-l' to enforce libtiff fallback...\n");
                        exit(ERR_NVCOMP_NOT_FOUND);
                }
        }
        fprintf(stderr, "trying libtiff...\n");
        return libtiff_decode(s->state_libtiff, fname);
}

static dec_image decode(struct state_gjtiff *s, const char *fname)
{
        INFO_MSG("\n%s"
               "==================================================================\n"
               "Processing input file %s...\n"
               "==================================================================%s\n",
                fg_bold, fname, term_reset);
        if (strstr(fname, ".jp2") == fname + strlen(fname) - 4) {
                return nvj2k_decode(s->state_nvj2k, fname);
        }
        return decode_tiff(s, fname);
}

struct ifiles {
        struct {
                char ifname[PATH_MAX];
                struct owned_image *img;
        } ifiles[3];
        int count;
};

static void ifiles_destroy(struct ifiles *ifiles) {
        for (int i = 0; i < ifiles->count; ++i) {
                if (ifiles->ifiles[i].img != nullptr) {
                        ifiles->ifiles[i].img->free(ifiles->ifiles[i].img);
                        ifiles->ifiles[i].img = nullptr;
                }
        }
}

static size_t encode_jpeg(struct state_gjtiff *s, struct dec_image uncomp,
                          size_t width_padding, const char *ofname)
{
        gpujpeg_parameters param = gpujpeg_default_parameters();
        param.interleaved = 1;
        if (s->opts.req_quality != -1) {
                param.quality = s->opts.req_quality;
        }
        if (log_level == LL_QUIET) {
                param.verbose = GPUJPEG_LL_QUIET;
        }
        if (log_level >= LL_DEBUG) {
                //  print out stats - only printed if verbose>=1 && perf_stats==1
                param.verbose = GPUJPEG_LL_DEBUG;
                param.perf_stats = 1;
        }

        int scaled_width = uncomp.width;
        int scaled_height = uncomp.height;
        struct owned_image *scaled = nullptr;
        if (gj_adjust_size(&scaled_width, &scaled_height, uncomp.comp_count)) {
                assert(width_padding == 0);
                scaled = scale(s->downscaler, scaled_width, scaled_height,
                               &uncomp);
                uncomp = scaled->img;
        }

        gpujpeg_image_parameters param_image =
            gpujpeg_default_image_parameters();
        param_image.width = uncomp.width;
        param_image.width_padding = width_padding;
        param_image.height = uncomp.height;
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 25, 0)
        param_image.comp_count = comp_count;
#endif
        param_image.color_space =
            uncomp.comp_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
        param_image.pixel_format = uncomp.comp_count == 1
                                       ? GPUJPEG_U8
                                       : GPUJPEG_444_U8_P012;
        gpujpeg_encoder_input encoder_input = gpujpeg_encoder_input_gpu_image(
            uncomp.data);
        uint8_t *out = nullptr;
        size_t len = 0;
        if (gpujpeg_encoder_encode(s->gj_enc, &param, &param_image, &encoder_input,
                               &out, &len) != 0) {
                ERROR_MSG("Failed to encode %s!\n", ofname);
        }

        if (len != 0) {
                FILE *outf = fopen(ofname, "wb");
                if (outf == nullptr) {
                        char buf[256];
                        ERROR_MSG("fopen %s: %s\n", ofname,
                                  strerror_r(errno, buf, sizeof buf));
                } else {
                        fwrite(out, len, 1, outf);
                        fclose(outf);
                }
        }
        if (scaled != nullptr) {
                scaled->free(scaled);
        }
        return len;
}

static void print_bbox(const struct coordinate coords[4]) {
        double lon_min = 0;
        double lon_max = 0;
        double lat_min = 0;
        double lat_max = 0;
        get_lat_lon_min_max(coords, &lat_min, &lat_max, &lon_min, &lon_max);
        if (lon_min < 0 && lon_max >= 0) { // handle antimeridian
                double tmp = lon_min;
                lon_min = lon_max;
                lon_max = tmp;
        }
        printf("\t\t\"bbox\": [%f, %f, %f, %f]\n", lon_min, lat_min, lon_max,
               lat_max);
}

static void get_ofname(const char *ifname, char *ofname, size_t buflen,
                       const char *suffix, char **endptr);

static struct owned_image *combine_images(const struct ifiles *ifiles,
                                          char *combined_ifname,
                                          cudaStream_t stream)
{
        assert(ifiles->count == 3);
        struct dec_image dst_desc = ifiles->ifiles[0].img->img;
        dst_desc.comp_count = 3;
        struct owned_image *ret = new_cuda_owned_image(&dst_desc);
        combine_images_cuda(&ret->img, &ifiles->ifiles[0].img->img,
                            &ifiles->ifiles[1].img->img,
                            &ifiles->ifiles[2].img->img, stream);

        // set new ifname
        combined_ifname[0] = '\0';
        char *const end = combined_ifname + PATH_MAX;
        get_ofname(ifiles->ifiles[0].ifname, combined_ifname,
                   end - combined_ifname, "", &combined_ifname);
        for (int i = 1; i < 3; ++i) {
                combined_ifname += snprintf(combined_ifname,
                                            end - combined_ifname, "-COMMA-");
                get_ofname(ifiles->ifiles[i].ifname, combined_ifname,
                           end - combined_ifname, "", &combined_ifname);
        }

        return ret;
}

/// uncomp->data must reside in device for JPEG, in RAM otherwise
static size_t encode_file(struct state_gjtiff *s,
                          const struct dec_image *uncomp, size_t width_padding,
                          const char *ofname, const struct dec_image *orig_img)
{
        switch (output_format) {
        case OUTF_JPEG:
                return encode_jpeg(s, *uncomp, width_padding, ofname);
        case OUTF_WEBP:
                return encode_webp(s->webp_enc, uncomp, width_padding, ofname, orig_img);
        case OUTF_RAW:
                return pam_write(ofname, uncomp->width,
                                 uncomp->width + width_padding, uncomp->height,
                                 uncomp->comp_count, 255, uncomp->data, true)
                           ? (size_t)uncomp->width * uncomp->comp_count *
                                 uncomp->height
                           : 0;
        }
        abort();
}

static struct owned_image *scale_webp(struct state_gjtiff *s,
                                       const struct dec_image *uncomp)
{
        int scaled_width = MAX_WEBP_DIMENSION;
        int scaled_height = MAX_WEBP_DIMENSION;
        if (uncomp->width > uncomp->height) {
                scaled_height = (int)((size_t)MAX_WEBP_DIMENSION *
                                      uncomp->height / uncomp->width);
        } else {
                scaled_width = (int)((size_t)MAX_WEBP_DIMENSION *
                                     uncomp->width / uncomp->height);
        }
        WARN_MSG("Encoding of %dx%d image exceeds max WebP "
                 "dimensions (16383x16383), downsizing to %dx%d!\n",
                 uncomp->width, uncomp->height, scaled_width, scaled_height);
        return scale(s->downscaler, scaled_width, scaled_height, uncomp);
}

///< encodes file in GPU memory
static size_t encode_gpu(struct state_gjtiff *s, const struct dec_image *uncomp,
                         const char *ofname)
{
        struct owned_image *in_ram = nullptr;
        if (output_format != OUTF_JPEG) {
                // handle WebP >= 16k*16k
                struct owned_image *scaled = nullptr;
                if (output_format == OUTF_WEBP &&
                    (uncomp->width > MAX_WEBP_DIMENSION ||
                     uncomp->height > MAX_WEBP_DIMENSION)) {
                        scaled = scale_webp(s, uncomp);
                        uncomp = &scaled->img;
                }

                in_ram = copy_img_from_device(uncomp, s->stream);
                uncomp = &in_ram->img;

                if (scaled != nullptr) {
                        scaled->free(scaled);
                }
        }
        const size_t len = encode_file(s, uncomp, 0, ofname, uncomp);
        if (in_ram != nullptr) {
                in_ram->free(in_ram);
        }
        return len;
}

static bool encode_single(struct state_gjtiff *s,
                   struct dec_image *uncomp, const char *ifname,
                   const char *ofname)
{
        const size_t len = encode_gpu(s, uncomp, ofname);
        char buf[UINT64_ASCII_LEN + 1];
        char fullpath[PATH_MAX + 1];
        realpath(ofname, fullpath);
        if (len != 0) {
                if (!s->first) {
                        printf(",\n");
                }
                printf("\t{\n");
                s->first = false;
                printf("\t\t\"infile\": \"%s\",\n", ifname);
                printf("\t\t\"outfile\": \"%s\",\n", fullpath);
                print_bbox(uncomp->coords);
                printf("\t}");
        }
        INFO_MSG("%s (%dx%d; %s B) encoded %ssuccessfully\n", ofname,
               uncomp->width, uncomp->height,
               format_number_with_delim(len, buf, sizeof buf),
               (len == 0 ? "un" : ""));
        return len != 0;
}

static char *get_tile_ofdir(const char *prefix, const char *ifname, int zoom, int x) {
        char *const ret = (char *) malloc(PATH_MAX);
        char *start = ret;
        char *const end = start + PATH_MAX;
        int written = snprintf(ret, end - start, "%s", prefix);
        start += written;
        get_ofname(ifname, start, end -  start, "", &start);
        mkdir(ret, 0755);
        start += snprintf(start, end - start, "/%d", zoom);
        mkdir(ret, 0755);
        start += snprintf(start, end - start, "/%d", x);
        mkdir(ret, 0755);
        return ret;
}

static bool encode_tiles_z(struct state_gjtiff *s,
                           struct dec_image *const uncomp, const char *ifname,
                           const char *prefix, int zoom_level)
{
        bool ret = true;
        const int scale = 1<<zoom_level;
        int x_first = floor(uncomp->bounds[XLEFT] * scale);
        int x_end = ceil(uncomp->bounds[XRIGHT] * scale);
        int dst_xpitch = x_end - x_first;
        int y_first = floor(uncomp->bounds[YTOP] * scale);
        int y_end = ceil(uncomp->bounds[YBOTTOM] * scale);
        int dst_lines = y_end - y_first;
        dst_xpitch *= 256 * uncomp->comp_count;
        dst_lines *= 256;
        /// @todo roundf below?
        int x_off = ((uncomp->bounds[XLEFT] * scale) - x_first) * 256.;
        int y_off = ((uncomp->bounds[YTOP] * scale) -
                 floor(uncomp->bounds[YTOP] * scale)) *
                256.;
        int new_height = ((uncomp->bounds[YBOTTOM] * scale) -
                          (uncomp->bounds[YTOP] * scale)) *
                         256.;
        int new_width = ((uncomp->bounds[XRIGHT] * scale) -
                          (uncomp->bounds[XLEFT] * scale)) *
                         256.;

        struct owned_image *scaled = scale_pitch(
            s->downscaler, new_width, x_off, dst_xpitch, new_height, y_off,
            dst_lines, uncomp, uncomp->width);
        const struct dec_image *src = &scaled->img;
        struct owned_image *in_ram = nullptr;
        if (output_format != OUTF_JPEG) {
                in_ram = copy_img_from_device(&scaled->img, s->stream);
                src = &in_ram->img;
        }

        // disable MT via OMP for JPEG
        omp_set_num_threads(output_format == OUTF_JPEG ? 1
                                                       : omp_get_max_threads());

        for (int x = 0; x < x_end - x_first; ++x) {
                char *path = get_tile_ofdir(prefix, ifname, zoom_level,
                                            x_first + x);
                const size_t path_len = strlen(path);
#pragma omp parallel for
                for (int y = 0; y < y_end - y_first; ++y) {
                        struct dec_image tile = scaled->img; // copy metadata
                        tile.width = tile.height = 256;
                        auto *fpath = (char *)malloc(PATH_MAX);
                        snprintf(fpath, PATH_MAX, "%s", path);
                        char *end = fpath + path_len;
                        snprintf(end, PATH_MAX - (end - fpath), "/%d%s",
                                 y_first + y, get_ext());
                        size_t off = ((ptrdiff_t)y * 256 * dst_xpitch) +
                                     (x * 256 * uncomp->comp_count);
                        tile.data = src->data + off;
                        if (src->alpha != nullptr) {
                                tile.alpha = src->alpha +
                                            off / uncomp->comp_count;
                        }
                        size_t len = encode_file(
                            s, &tile, dst_xpitch - 256 * uncomp->comp_count,
                            fpath, src);
                        if (len != 0) {
                                // printf(", \"%s\"", path);
                        } else {
                                ret = false;
                        }
                        free(fpath);
                }
                free(path);
        }
        scaled->free(scaled);
        if (in_ram != nullptr) {
                in_ram->free(in_ram);
        }
        return ret;
}

static bool encode_tiles(struct state_gjtiff *s, struct dec_image *const uncomp,
                         const char *ifname, const char *prefix,
                         int *zoom_levels)
{
        bool ret = true;
        char whole[PATH_MAX];
        snprintf(whole, sizeof whole, "%s", prefix);
        get_ofname(ifname, whole + strlen(whole), sizeof whole - strlen(whole),
                   get_ext(), nullptr);
        if (!s->opts.no_whole_image) {
                if (encode_gpu(s, uncomp, whole) == 0) {
                        return false;
                }
        }
        if (!s->first) {
                printf(",\n");
        }
        printf("\t{\n");
        s->first = false;
        printf("\t\t\"infile\":\"%s\",\n", ifname);
        printf("\t\t\"outfile\":");
        printf("\"%s\"", whole);
        while (*zoom_levels != -1) {
                ret &= encode_tiles_z(s, uncomp, ifname, prefix, *zoom_levels);
                zoom_levels++;
        }
        printf("\n");
        // printf(",\n"); print_bbox(uncomp->coords);
        printf("\t}");
        INFO_MSG("%s encoded %ssuccessfully\n", whole,
               (ret ? "" : "un"));
        return ret;
}

static void get_ofname(const char *ifname, char *ofname, size_t buflen,
                       const char *suffix, char **endptr)
{
        buflen = std::min<size_t>(buflen, NAME_MAX + 1);

        char *basename = nullptr;
        if (strrchr(ifname, '/') != nullptr) {
                basename = strdup(strrchr(ifname, '/') + 1);
        } else {
                basename = strdup(ifname);
        }
        char *last_dot = strrchr(basename, '.');
        if (last_dot != nullptr) {
                *last_dot = '\0';
        }
        int written = snprintf(ofname, buflen, "%s%s", basename, suffix);
        if (endptr != nullptr) {
                *endptr = ofname + written;
        }
        free(basename);
}

static void show_help(const char *progname)
{
        INFO_MSG("%s [options] img [img2...]\n", progname);
        INFO_MSG("%s [options] -\n\n", progname);
        INFO_MSG("Options:\n");
        INFO_MSG("\t-d       - list of CUDA devices\n");
        INFO_MSG("\t-h       - show help\n");
        INFO_MSG("\t-l       - use libtiff if nvCOMP not available\n");
        INFO_MSG("\t-n       - do not adjust to natural rotation/prooprotion\n");
        INFO_MSG("\t-r       - write raw PNM instead of JPEG\n");
        INFO_MSG("\t-w       - WebP encode\n");
        INFO_MSG("\t-o <dir> - output JPEG directory\n");
        INFO_MSG("\t-q <q>   - JPEG quality\n");
        INFO_MSG("\t-s <d>   - downscale factor\n");
        INFO_MSG("\t-v[v]    - be verbose (2x for more messages)\n");
        INFO_MSG("\t-Q[Q]    - be quiet (do not print anything except produced files), double to suppress also warnings\n");
        INFO_MSG("\t-I <num> - downsampling interpolation idx (NppiInterpolationMode; default 8 /SUPER/)\n");
        INFO_MSG("\t-M <sz_GB>- GPUJPEG memory limit (in GB, floating point; default 1/2 of available VRAM)\n");
        INFO_MSG("\t-z <zlevel>- zoom level\n");
        INFO_MSG("\t-V       - print version (Git commit)\n");
        INFO_MSG("\n");
        INFO_MSG("Input must be in TIFF or JP2.\"\n");
        INFO_MSG("Output filename will be \"basename ${name%%.*}.jpg\"\n");
        INFO_MSG("Output directory must exist, implicitly \".\"\n\n");
        INFO_MSG("If the '-' is given as an argument, newline-separated list of "
               "file "
               "names\nis read from stdin.\n");
        INFO_MSG("\n");
        INFO_MSG("Input filename (both cmdline argument or from pipe) can be suffixed with opts,\n");
        INFO_MSG("syntax:\n");
        INFO_MSG("\tfname[:q=<JPEG_q>][:s=<downscale_factor>]\n");
}

static char *parse_fname_opts(char *buf, struct options *opts)
{
        if (buf == nullptr) {
                return nullptr;
        }
        char *save_ptr = nullptr;
        char *fname = strtok_r(buf, ":", &save_ptr);
        char *item = nullptr;
        while ((item = strtok_r(nullptr, ":", &save_ptr)) != nullptr) {
                if (strstr(item, "q=") == item) {
                        opts->req_quality = (int)strtol(
                            strchr(item, '=') + 1, nullptr, 10);
                } else if (strstr(item, "s=") == item) {
                        opts->downscale_factor = (int)strtol(
                            strchr(item, '=') + 1, nullptr, 10);
                } else {
                        ERROR_MSG("Wrong option: %s!\n", item);
                }
        }
        return fname;
}

/// @returns filename to process either from argv or read from stdin
static char *get_next_ifname(bool from_stdin, char ***argv, char *buf,
                             size_t buflen, struct options *opts)
{
        if (!from_stdin) {
                return parse_fname_opts(*(*argv)++, opts);
        }
        char *ret = fgets(buf, buflen, stdin);
        if (ret == nullptr) {
                return ret;
        }
        // trim NL
        const size_t line_len = strlen(buf);
        if (buf[line_len - 1] == '\n') {
                buf[line_len - 1] = '\0';
        }
        return parse_fname_opts(buf, opts);
}

static ifiles parse_ifiles(const char *ifnames)
{
        struct ifiles ret = {};
        char copy[PATH_MAX];
        snprintf(copy, sizeof copy, "%s", ifnames);
        char *saveptr = nullptr;
        char *item = nullptr;
        char *tmp = copy;
        while ((item = strtok_r(tmp, ",", &saveptr)) != nullptr) {
                if (ret.count == ARR_SIZE(ret.ifiles)) {
                        ERROR_MSG("More than 3 images not supported!\n");
                        return {};
                }
                snprintf(ret.ifiles[ret.count++].ifname,
                         sizeof ret.ifiles[0].ifname, "%s", item);
                tmp = nullptr;
        }
        if (ret.count == 2) {
                ERROR_MSG("Combination of 2 bands unsupported, must be 3!\n");
                return {};
        }
        return ret;
}


static void parse_zoom_levels(int *zoom_levels, char *optarg) {
        char *item = nullptr;
        char *save_ptr = nullptr;
        int idx = 0;
        while ((item = strtok_r(optarg, ",", &save_ptr)) != nullptr) {
                optarg = nullptr;
                assert(idx < MAX_ZOOM_COUNT - 1);
                zoom_levels[idx++] = (int)strtol(item, nullptr, 0);
        }
        zoom_levels[idx++] = -1;
}

const char *fg_bold = "";
const char *fg_red = "";
const char *fg_yellow = "";
const char *term_reset = "";

static void init_term_colors() {
        if (isatty(fileno(stdout)) && isatty(fileno(stderr))) {
                fg_bold = "\033[1m";
                fg_red = "\033[31m";
                fg_yellow = "\033[33m";
                term_reset = "\033[0m";
        }
}

int main(int argc, char **argv)
{
        init_term_colors();

        char ofdir[PATH_MAX] = "./";
        struct options global_opts = OPTIONS_INIT;

        int opt = 0;
        while ((opt = getopt(argc, argv, "+I:M:NQVdhnno:q:rs:vwz:")) != -1) {
                switch (opt) {
                case 'I':
                        interpolation = (int)strtol(optarg, nullptr, 0);
                        break;
                case 'M':
                        mem_limit = strtof(optarg, nullptr) * 1E9;
                        break;
                case 'Q':
                        log_level -= 1;
                        break;
                case 'V':
                        puts(GIT_REV_STR);
                        return 0;
                case 'd':
                        return !!gpujpeg_print_devices_info();
                case 'h':
                        show_help(argv[0]);
                        return EXIT_SUCCESS;
                case 'l':
                        global_opts.use_libtiff = true;
                        break;
                case 'n':
                        global_opts.norotate = true;
                        break;
                case 'N':
                        global_opts.no_whole_image = true;
                        break;
                case 'o':
                        snprintf(ofdir, sizeof ofdir, "%s/", optarg);
                        break;
                case 'q':
                        global_opts.req_quality = (int)strtol(
                            optarg, nullptr, 10);
                        break;
                case 'r':
                        output_format = OUTF_RAW;
                        break;
                case 's':
                        global_opts.downscale_factor = (int)strtol(optarg,
                                                                   nullptr, 10);
                        break;
                case 'v':
                        log_level += 1;
                        break;
                case 'w':
                        output_format = OUTF_WEBP;
                        break;
                case 'z':
                        parse_zoom_levels(global_opts.zoom_levels, optarg);
                        break;
                default: /* '?' */
                        show_help(argv[0]);
                        exit(EXIT_FAILURE);
                }
        }

        if (optind == argc) {
                show_help(argv[0]);
                return EXIT_FAILURE;
        }

        gpu_memory = get_cuda_dev_global_memory();
        struct state_gjtiff state(global_opts);
        int ret = EXIT_SUCCESS;

        char path_buf[PATH_MAX];
        argv += optind;
        const bool fname_from_stdin = strcmp(argv[0], "-") == 0;
        const size_t d_pref_len = strlen(ofdir);
        struct options opts = global_opts;

        printf("[\n");

        while (char *ifname = get_next_ifname(fname_from_stdin, &argv, path_buf,
                                              sizeof path_buf, &opts)) {
                struct ifiles ifiles = parse_ifiles(ifname);
                if (ifiles.count == 0) {
                        ret = EXIT_FAILURE;
                        continue;
                }
                TIMER_START(transcode, LL_VERBOSE);
                bool err = false;
                for (int i = 0; i < ifiles.count; ++i) {
                        struct dec_image dec = decode(&state, ifiles.ifiles[i].ifname);
                        if (dec.data == nullptr) {
                                ret = ERR_SOME_FILES_NOT_TRANSCODED;
                                err = true;
                                break;
                        }
                        if (dec.comp_count != 1 && dec.comp_count != 3) {
                                ERROR_MSG(
                                    "Only 1 or 3 channel images are currently "
                                    "supported! Skipping %s...\n",
                                    ifname);
                                ret = ERR_SOME_FILES_NOT_TRANSCODED;
                                err = true;
                                break;
                        }
                        if (opts.downscale_factor != 1) {
                                dec = downscale(state.downscaler,
                                                opts.downscale_factor, &dec);
                        }
                        dec.alpha = nullptr;
                        ifiles.ifiles[i].img = rotate(state.rotate, &dec);
                        assert(ifiles.ifiles[i].img != nullptr);
                }
                if (err) {
                        goto skip;
                }
                if (ifiles.count > 1) {
                        char combined_ifname[PATH_MAX];
                        struct owned_image *combined = combine_images(
                            &ifiles, combined_ifname, state.stream);
                        ifiles_destroy(&ifiles);
                        ifiles.ifiles[0].img = combined;
                        snprintf(ifiles.ifiles[0].ifname,
                                 sizeof ifiles.ifiles[0].ifname, "%s",
                                 combined_ifname);
                        ifiles.count = 1;
                }
                if (global_opts.zoom_levels[0] == -1) {
                        get_ofname(ifiles.ifiles[0].ifname, ofdir + d_pref_len,
                                   sizeof ofdir - d_pref_len, get_ext(),
                                   nullptr);
                        ret = encode_single(&state, &ifiles.ifiles[0].img->img,
                                            ifname, ofdir)
                                  ? ret
                                  : ERR_SOME_FILES_NOT_TRANSCODED;
                } else {
                        ret = encode_tiles(&state, &ifiles.ifiles[0].img->img,
                                           ifiles.ifiles[0].ifname, ofdir,
                                           global_opts.zoom_levels)
                                  ? ret
                                  : ERR_SOME_FILES_NOT_TRANSCODED;
                }
        skip:
                ifiles_destroy(&ifiles);
                TIMER_STOP(transcode);
        }

        cleanup_cuda_kernels();

        printf("\n]\n");

        return ret;
}
