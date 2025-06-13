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

struct state_gjtiff {
        state_gjtiff(bool use_libtiff, bool norotate, bool write_uncompressed);
        ~state_gjtiff();
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

state_gjtiff::state_gjtiff(bool u, bool norotate, bool write_uncompressed)
    : use_libtiff(u)
{
        CHECK_CUDA(cudaStreamCreate(&stream));
        CHECK_CUDA(cudaEventCreate(&cuda_event_start));
        CHECK_CUDA(cudaEventCreate(&cuda_event_stop));

        state_libtiff = libtiff_create(log_level, stream);
        state_nvj2k = nvj2k_init(stream);
        assert(state_nvj2k != nullptr);
        state_nvtiff = nvtiff_init(stream, log_level);
        assert(state_nvtiff != nullptr);
        if (!write_uncompressed) {
                gj_enc = gpujpeg_encoder_create(stream);
                assert(gj_enc != nullptr);
        }
        downscaler = downscaler_init(stream);
        assert(downscaler != nullptr);
        if (!norotate) {
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

static size_t encode_jpeg(struct state_gjtiff *s, int req_quality, struct dec_image uncomp,
                        const char *ofname)
{
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
        param_image.width = uncomp.width;
        param_image.height = uncomp.height;
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 25, 0)
        param_image.comp_count = comp_count;
#endif
        param_image.color_space =
            uncomp.comp_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
        param_image.pixel_format =
            uncomp.comp_count == 1 ? GPUJPEG_U8 : GPUJPEG_444_U8_P012;
        gpujpeg_encoder_input encoder_input = gpujpeg_encoder_input_gpu_image(
            uncomp.data);
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

static void print_bbox(struct coordinate coords[4]) {
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

static void encode(struct state_gjtiff *s, int req_quality,
                   struct dec_image uncomp, const char *ifname,
                   const char *ofname)
{
        size_t len = 0;
        if (s->gj_enc != nullptr) {
                len = encode_jpeg(s, req_quality, uncomp, ofname);
        } else {
                len = uncomp.width * uncomp.height * uncomp.comp_count;
                unsigned char *data = new unsigned char[len];
                CHECK_CUDA(cudaMemcpy(data, uncomp.data, len, cudaMemcpyDefault));
                    pam_write(ofname, uncomp.width, uncomp.height,
                              uncomp.comp_count, 255, data, true);
                delete[] data;
        }
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
                print_bbox(uncomp.coords);
                printf("\t}");
        }
        INFO_MSG("%s (%dx%d; %s B) encoded %ssuccessfully\n", ofname,
               uncomp.width, uncomp.height,
               format_number_with_delim(len, buf, sizeof buf),
               (len == 0 ? "un" : ""));
}

static void set_ofname(const char *ifname, char *ofname, size_t buflen, bool jpeg)
{
        const char *ext = jpeg ? "jpg" : "pnm";

        if (strrchr(ifname, '/') != nullptr) {
                snprintf(ofname, buflen, "%s", strrchr(ifname, '/') + 1);
        } else {
                snprintf(ofname, buflen, "%s", ifname);
        }
        if (strrchr(ofname, '.') != nullptr) {
                char *ptr = strrchr(ofname, '.') + 1;
                size_t avail_len = buflen - (ptr - ofname);
                snprintf(ptr, avail_len, "%s", ext);
        } else {
                snprintf(ofname + strlen(ofname), buflen - strlen(ofname),
                         ".%s", ext);
        }
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

struct options {
        int req_gpujpeg_quality;
        int downscale_factor;
#define OPTIONS_INIT {-1, 1}
};

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

        bool use_libtiff = false;
        bool norotate = false;
        bool write_uncompressed = false;
        char ofdir[1024] = "./";
        struct options global_opts = OPTIONS_INIT;

        int opt = 0;
        while ((opt = getopt(argc, argv, "+I:M:Qdhnno:q:rs:v")) != -1) {
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
                case 'd':
                        return !!gpujpeg_print_devices_info();
                case 'h':
                        show_help(argv[0]);
                        return EXIT_SUCCESS;
                case 'l':
                        use_libtiff = true;
                        break;
                case 'n':
                        norotate = true;
                        break;
                case 'o':
                        snprintf(ofdir, sizeof ofdir, "%s/", optarg);
                        break;
                case 'q':
                        global_opts.req_gpujpeg_quality = (int)strtol(
                            optarg, nullptr, 10);
                        break;
                case 'r':
                        write_uncompressed = true;
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
        struct state_gjtiff state(use_libtiff, norotate, write_uncompressed);
        int ret = EXIT_SUCCESS;

        char path_buf[PATH_MAX];
        argv += optind;
        const bool fname_from_stdin = strcmp(argv[0], "-") == 0;
        const size_t d_pref_len = strlen(ofdir);
        struct options opts = global_opts;

        printf("[\n");

        while (char *ifname = get_next_ifname(fname_from_stdin, &argv, path_buf,
                                              sizeof path_buf, &opts)) {
                TIMER_START(transcode, LL_VERBOSE);
                set_ofname(ifname, ofdir + d_pref_len,
                           sizeof ofdir - d_pref_len, state.gj_enc != nullptr);

                struct dec_image dec = decode(&state, ifname);
                if (dec.data == nullptr) {
                        ret = ERR_SOME_FILES_NOT_TRANSCODED;
                        continue;
                }
                if (dec.comp_count != 1 && dec.comp_count != 3) {
                        ERROR_MSG("Only 1 or 3 channel images are currently "
                                  "supported! Skipping %s...\n",
                                  ifname);
                        ret = ERR_SOME_FILES_NOT_TRANSCODED;
                        continue;
                }
                if (opts.downscale_factor != 1) {
                        dec = downscale(state.downscaler,
                                        opts.downscale_factor, &dec);
                }
                dec = rotate(state.rotate, &dec);
                encode(&state, opts.req_gpujpeg_quality, dec, ifname, ofdir);
                TIMER_STOP(transcode);
        }

        cleanup_cuda_kernels();

        printf("\n]\n");

        return ret;
}
