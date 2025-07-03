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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <libgpujpeg/gpujpeg_type.h>
#include <libgpujpeg/gpujpeg_version.h>
#include <linux/limits.h>
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

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

int log_level = 0;
size_t gpu_memory = 0;
cudaEvent_t cuda_event_start;
cudaEvent_t cuda_event_stop;

int interpolation = 0;
long long mem_limit = 0;

struct options {
        int req_gpujpeg_quality;
        int downscale_factor;
        bool use_libtiff;
        bool norotate;
        bool write_uncompressed;
        bool debug;
#define OPTIONS_INIT {-1, 1, false, false, false, false}
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
        // downscaler
        struct downscaler_state *downscaler = NULL;
        // rotate
        struct rotate_state *rotate = NULL;

        bool first = true;
};

state_gjtiff::state_gjtiff(struct options opts)
    : opts(opts)
{
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaEventCreate(&cuda_event_start));
        CHECK_CUDA(cudaEventCreate(&cuda_event_stop));

        state_libtiff = libtiff_create(log_level, stream);
        state_nvj2k = nvj2k_init(stream);
        assert(state_nvj2k != nullptr);
        state_nvtiff = nvtiff_init(stream, log_level);
        assert(state_nvtiff != nullptr);
        if (!opts.write_uncompressed) {
                gj_enc = gpujpeg_encoder_create(stream);
                assert(gj_enc != nullptr);
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
        nvj2k_destroy(state_nvj2k);
        nvtiff_destroy(state_nvtiff);
        libtiff_destroy(state_libtiff);
        downscaler_destroy(downscaler);
        rotate_destroy(rotate);

        CHECK_CUDA(cudaEventDestroy(cuda_event_start));
        CHECK_CUDA(cudaEventDestroy(cuda_event_stop));
        // destroy last - components may hold the stream
        CHECK_CUDA(cudaStreamDestroy(stream));

        if (opts.debug) {
                cudaDeviceReset();
        }
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

struct ifile {
        char ifname[PATH_MAX];
        struct owned_image *img;
        struct ifile *next;
};
struct ifiles {
        struct ifile *head;
};

static struct ifiles *new_ifiles(struct owned_image *head)
{
        struct ifiles *ret = (struct ifiles *)calloc(1, sizeof *ret);
        if (head == NULL) {
                return ret;
        }
        struct ifile *ifile = (struct ifile *)calloc(1, sizeof *ifile);
        ret->head = ifile;
        ret->head->img = head;
        return ret;
}

static void ifile_destroy(struct ifile *ifile)
{
        if (ifile->img == nullptr) {
                return;
        }
        ifile->img->free(ifile->img);
        ifile->img = nullptr;
}

static void ifiles_destroy(struct ifiles **ifiles) {
        if (*ifiles == nullptr) {
                return;
        }
        struct ifile *cur = (*ifiles)->head;
        while (cur != nullptr) {
                ifile_destroy(cur);
                struct ifile *tmp = cur;
                cur = cur->next;
                free(tmp);
        }
        free(*ifiles);
        *ifiles = nullptr;
}
static size_t encode_jpeg(struct state_gjtiff *s, int req_quality,
                          const struct owned_image *img, const char *ofname)
{
        const struct dec_image *uncomp = &img->img;
        gpujpeg_parameters param = gpujpeg_default_parameters();
        param.interleaved = 1;
        if (req_quality != -1) {
                param.quality = req_quality;
        }
        if (log_level == LL_QUIET) {
                param.verbose = GPUJPEG_LL_QUIET;
        }
        if (log_level >= LL_DEBUG) {
                //  print out stats - only printed if verbose>=1 && perf_stats==1
                param.verbose = GPUJPEG_LL_DEBUG;
                param.perf_stats = 1;
        }

        gpujpeg_image_parameters param_image =
            gpujpeg_default_image_parameters();
        param_image.width = uncomp->width;
        param_image.height = uncomp->height;
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 25, 0)
        param_image.comp_count = comp_count;
#endif
        param_image.color_space =
            uncomp->comp_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
        param_image.pixel_format = uncomp->comp_count == 1
                                       ? GPUJPEG_U8
                                       : (img->planar ? GPUJPEG_444_U8_P0P1P2
                                                 : GPUJPEG_444_U8_P012);
        gpujpeg_encoder_input encoder_input = gpujpeg_encoder_input_gpu_image(
            uncomp->data);
        uint8_t *out = nullptr;
        size_t len = 0;
        if (gpujpeg_encoder_encode(s->gj_enc, &param, &param_image, &encoder_input,
                               &out, &len) != 0) {
                ERROR_MSG("Failed to encode %s!\n", ofname);
                return 0;
        }

        FILE *outf = fopen(ofname, "wb");
        if (outf == nullptr) {
                ERROR_MSG("fopen %s: %s\n", ofname, strerror(errno));
                return 0;
        }
        fwrite(out, len, 1, outf);
        fclose(outf);
        return len;
}

static void print_bbox(struct coordinate const *coords) {
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

static ifiles *unify_sizes(struct state_gjtiff *s,
                                   struct ifiles **ifiles)
{
        const struct dec_image *first = &(*ifiles)->head->img->img;
        struct ifile *cur = (*ifiles)->head;
        int count = 0;
        while (cur != nullptr) {
                // safety checks
                if (cur->img->img.comp_count != 1) {
                        ERROR_MSG("Cannot combine image with %d channels!!\n",
                                  cur->img->img.comp_count);
                        ifiles_destroy(ifiles);
                        return nullptr;
                }
                if (++count > 3) {
                        ERROR_MSG("Cannot combine more than 3 images!!\n");
                        ifiles_destroy(ifiles);
                        return nullptr;
                }

                if (cur->img->img.width != first->width ||
                    cur->img->img.height != first->height) {
                        VERBOSE_MSG("Incompatible size %dx%d vs %dx%d, scaling "
                                    "to the later one...\n",
                                    cur->img->img.width, cur->img->img.height,
                                    first->width, first->height);
                        struct owned_image *scaled = scale(
                            s->downscaler, first->width, first->height,
                            cur->img);
                        // remove delete old data
                        cur->img->free(cur->img);
                        // set the scaled one, instead
                        cur->img = scaled;
                }
                cur = cur->next;
        }
        return *ifiles;
}

static ifiles *combine_images(struct state_gjtiff *s,
                                   struct ifiles **ifiles)
{
        const size_t plane_lenght = (size_t)(*ifiles)->head->img->img.width *
                                    (*ifiles)->head->img->img.height;
        struct dec_image new_desc = (*ifiles)->head->img->img;
        new_desc.comp_count = 3;
        struct owned_image *nimg = new_cuda_owned_image(&new_desc);
        nimg->planar = true;
        uint8_t *d_ptr = nimg->img.data;
        struct ifile *cur = (*ifiles)->head;
        while (cur != nullptr) {
                CHECK_CUDA(cudaMemcpyAsync(d_ptr,
                                           cur->img->img.data, plane_lenght,
                                           cudaMemcpyDefault, s->stream));
                cur = cur->next;
                d_ptr += plane_lenght;
        }
        size_t img_len = d_ptr - nimg->img.data;
        if (img_len < 3 * plane_lenght) {
                CHECK_CUDA(cudaMemsetAsync(d_ptr, 0, (3 * plane_lenght) - img_len,
                                           s->stream));
        }
        ifiles_destroy(ifiles);
        return new_ifiles(nimg);
}

static ifiles *process_images(struct state_gjtiff *s,
                                   struct ifiles **ifiles) {
        if ((*ifiles)->head->next == nullptr) {
                return *ifiles;
        }
        *ifiles = unify_sizes(s, ifiles);
        if (*ifiles == nullptr) {
                return nullptr;
        }
        *ifiles = combine_images(s, ifiles);
        if (*ifiles == nullptr) {
                return nullptr;
        }
        return *ifiles;
}

static void encode(struct state_gjtiff *s, int req_quality,
                   struct ifiles **ifiles, const char *ifname,
                   const char *ofname)
{
        *ifiles = process_images(s, ifiles);
        if (*ifiles == nullptr) {
                return;
        }
        size_t len = 0;
        const struct owned_image *img = (*ifiles)->head->img;
        const struct dec_image *uncomp = &img->img;
        if (s->gj_enc != nullptr) {
                len = encode_jpeg(s, req_quality, img,
                                  ofname);
        } else if (!img->planar) {
                len = (size_t)uncomp->width * uncomp->height *
                      uncomp->comp_count;
                unsigned char *data = new unsigned char[len];
                CHECK_CUDA(cudaMemcpy(data, uncomp->data, len, cudaMemcpyDefault));
                    pam_write(ofname, uncomp->width, uncomp->height,
                              uncomp->comp_count, 255, data, true);
                delete[] data;
        } else {
                ERROR_MSG("Cannot write planar RAW!\n");
        }
        ifiles_destroy(ifiles);
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
}



static void set_ofname(const struct ifiles *ifiles, char *ofname, size_t buflen, bool jpeg)
{
        const char *ext = jpeg ? "jpg" : "pnm";
        buflen = std::min<size_t>(buflen, NAME_MAX + 1);
        assert(buflen >= 5);
        buflen -= 4;  // reserve 4 B for .ext

        struct ifile *cur = ifiles->head;
        while (cur != nullptr) {
                const char *ifname = cur->ifname;
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
                char delim[4] = "";
                if (cur != ifiles->head) {
                        snprintf(delim, sizeof delim, "%%%X", ',');
                }
                int bytes = snprintf(ofname, buflen, "%s%s", delim, basename);
                if (bytes >= (int) buflen) { // output truncated
                        bytes = (int) buflen - 1;
                }
                ofname += bytes;
                buflen -= bytes;
                free(basename);
                cur = cur->next;
        }
        snprintf(ofname, 5, ".%s", ext);
}

static void show_help(const char *progname)
{
        INFO_MSG("%s [options] img [img2...]\n", progname);
        INFO_MSG("%s [options] -\n\n", progname);
        INFO_MSG("Options:\n");
        INFO_MSG("\t-b       - debug (currently only emits cudaDeviceReset())\n");
        INFO_MSG("\t-d       - list of CUDA devices\n");
        INFO_MSG("\t-h       - show help\n");
        INFO_MSG("\t-l       - use libtiff if nvCOMP not available\n");
        INFO_MSG("\t-n       - do not adjust to natural rotation/prooprotion\n");
        INFO_MSG("\t-r       - write raw PNM instead of JPEG\n");
        INFO_MSG("\t-o <dir> - output JPEG directory\n");
        INFO_MSG("\t-q <q>   - JPEG quality\n");
        INFO_MSG("\t-s <d>   - downscale factor\n");
        INFO_MSG("\t-v[v]    - be verbose (2x for more messages)\n");
        INFO_MSG("\t-Q[Q]    - be quiet (do not print anything except produced files), double to suppress also warnings\n");
        INFO_MSG("\t-I <num> - downsampling interpolation idx (NppiInterpolationMode; default 8 /SUPER/)\n");
        INFO_MSG("\t-M <sz_GB>- GPUJPEG memory limit (in GB, floating point; default 1/2 of available VRAM)\n");
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
                        opts->req_gpujpeg_quality = (int)strtol(
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

static ifiles *parse_ifiles(const char *ifnames)
{
        struct ifiles *ret = new_ifiles(nullptr);
        struct ifile **tail = &ret->head;
        char copy[PATH_MAX];
        snprintf(copy, sizeof copy, "%s", ifnames);
        char *saveptr = nullptr;
        char *item = nullptr;
        char *tmp = copy;
        while ((item = strtok_r(tmp, ",", &saveptr)) != nullptr) {
                *tail = (struct ifile *)calloc(1, sizeof **tail);
                snprintf((*tail)->ifname, sizeof(*tail)->ifname, "%s", item);
                tail = &(*tail)->next;
                tmp = nullptr;
        }
        return ret;
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
        while ((opt = getopt(argc, argv, "+I:M:Qbdhnno:q:rs:v")) != -1) {
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
                case 'b':
                        global_opts.debug = true;
                        break;
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
                case 'o':
                        snprintf(ofdir, sizeof ofdir, "%s/", optarg);
                        break;
                case 'q':
                        global_opts.req_gpujpeg_quality = (int)strtol(
                            optarg, nullptr, 10);
                        break;
                case 'r':
                        global_opts.write_uncompressed = true;
                        break;
                case 's':
                        global_opts.downscale_factor = (int)strtol(optarg,
                                                                   nullptr, 10);
                        break;
                case 'v':
                        log_level += 1;
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
                struct ifiles *ifiles = parse_ifiles(ifname);
                if (ifiles == nullptr) {
                        ret = EXIT_FAILURE;
                        continue;
                }
                TIMER_START(transcode, LL_VERBOSE);
                bool err = false;
                struct ifile *cur = ifiles->head;
                while (cur != nullptr) {
                        struct dec_image dec = decode(&state, cur->ifname);
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
                        cur->img = rotate(state.rotate, &dec);
                        cur = cur->next;
                }
                if (!err) {
                        set_ofname(ifiles, ofdir + d_pref_len,
                                   sizeof ofdir - d_pref_len,
                                   state.gj_enc != nullptr);
                        encode(&state, opts.req_gpujpeg_quality, &ifiles,
                               ifname, ofdir);
                }
                ifiles_destroy(&ifiles);
                TIMER_STOP(transcode);
        }

        cleanup_cuda_kernels();

        printf("\n]\n");

        return ret;
}
