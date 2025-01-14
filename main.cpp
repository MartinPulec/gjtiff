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
#include "utils.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

int log_level = 0;

struct state_gjtiff {
        state_gjtiff(bool use_libtiff);
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
};

state_gjtiff::state_gjtiff(bool u)
    : use_libtiff(u)
{
        CHECK_CUDA(cudaStreamCreate(&stream));
        state_libtiff = libtiff_create(log_level, stream);
        state_nvj2k = nvj2k_init(stream);
        state_nvtiff = nvtiff_init(stream, log_level);
        assert(state_nvtiff != nullptr);
        gj_enc = gpujpeg_encoder_create(stream);
        assert(gj_enc != nullptr);
        downscaler = downscaler_init(stream);
        assert(downscaler != nullptr);
}

state_gjtiff::~state_gjtiff()
{
        gpujpeg_encoder_destroy(gj_enc);
        nvj2k_destroy(state_nvj2k);
        nvtiff_destroy(state_nvtiff);
        libtiff_destroy(state_libtiff);
        downscaler_destroy(downscaler);
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
        printf("Decoding from file %s... \n", fname);
        if (strstr(fname, ".jp2") == fname + strlen(fname) - 4) {
                return nvj2k_decode(s->state_nvj2k, fname);
        }
        return decode_tiff(s, fname);
}

static void encode_jpeg(struct state_gjtiff *s, int req_quality, struct dec_image uncomp,
                        const char *ofname)
{
        gpujpeg_parameters param = gpujpeg_default_parameters();
        if (req_quality != -1) {
                param.quality = req_quality;
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
                return;
        }

        if (log_level >= 1) {
                printf("%s encoded successfully\n", ofname);
        }
        FILE *outf = fopen(ofname, "wb");
        if (outf == nullptr) {
                ERROR_MSG("fopen %s: %s\n", ofname, strerror(errno));
                return;
        }
        fwrite(out, len, 1, outf);
        fclose(outf);
}

static void set_ofname(const char *ifname, char *ofname, size_t buflen)
{
        if (strrchr(ifname, '/') != nullptr) {
                snprintf(ofname, buflen, "%s", strrchr(ifname, '/') + 1);
        } else {
                snprintf(ofname, buflen, "%s", ifname);
        }
        if (strrchr(ofname, '.') != nullptr) {
                char *ptr = strrchr(ofname, '.') + 1;
                size_t avail_len = buflen - (ptr - ofname);
                snprintf(ptr, avail_len, "jpg");
        } else {
                snprintf(ofname + strlen(ofname), buflen - strlen(ofname),
                         ".jpg");
        }
}

static void show_help(const char *progname)
{
        printf("%s [options] img1.tif [img2.tif...]\n", progname);
        printf("%s [options] -\n\n", progname);
        printf("Options:\n");
        printf("\t-d       - list of CUDA devices\n");
        printf("\t-h       - show help\n");
        printf("\t-l       - use libtiff if nvCOMP not available\n");
        printf("\t-o <dir> - output JPEG directory\n");
        printf("\t-q <q>   - JPEG quality\n");
        printf("\t-s <d>   - downscale factor\n");
        printf("\t-v[v]    - be verbose (2x for more messages)\n");
        printf("\n");
        printf("Output filename will be \"basename ${name%%.*}.jpg\"\n");
        printf("Output directory must exist, implicitly \".\"\n\n");
        printf("If the '-' is given as an argument, newline-separated list of "
               "file "
               "names\nis read from stdin.\n");
        printf("\n");
        printf("Input filename (both cmdline argument or from pipe) can be suffixed with opts,\n");
        printf("syntax:\n");
        printf("\tfname[:q=<JPEG_q>][:s=<downscale_factor>]\n");
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

const char *fg_red = "";
const char *fg_yellow = "";
const char *term_reset = "";

static void init_term_colors() {
        if (isatty(fileno(stdout)) && isatty(fileno(stderr))) {
                fg_red = "\033[31m";
                fg_yellow = "\033[33m";
                term_reset = "\033[0m";
        }
}

int main(int argc, char **argv)
{
        init_term_colors();

        bool use_libtiff = false;
        char ofdir[1024] = "./";
        struct options global_opts = OPTIONS_INIT;

        int opt = 0;
        while ((opt = getopt(argc, argv, "+dhlo:q:s:v")) != -1) {
                switch (opt) {
                case 'd':
                        return !!gpujpeg_print_devices_info();
                case 'h':
                        show_help(argv[0]);
                        return EXIT_SUCCESS;
                case 'l':
                        use_libtiff = true;
                        break;
                case 'o':
                        snprintf(ofdir, sizeof ofdir, "%s/", optarg);
                        break;
                case 'q':
                        global_opts.req_gpujpeg_quality = (int)strtol(
                            optarg, nullptr, 10);
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

        struct state_gjtiff state(use_libtiff);
        int ret = EXIT_SUCCESS;

        char path_buf[PATH_MAX];
        argv += optind;
        const bool fname_from_stdin = strcmp(argv[0], "-") == 0;
        const size_t d_pref_len = strlen(ofdir);
        struct options opts = global_opts;
        while (char *ifname = get_next_ifname(fname_from_stdin, &argv, path_buf,
                                              sizeof path_buf, &opts)) {
                TIMER_START(transcode, LL_VERBOSE);
                set_ofname(ifname, ofdir + d_pref_len,
                           sizeof ofdir - d_pref_len);

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
                encode_jpeg(&state, opts.req_gpujpeg_quality, dec, ofdir);
                TIMER_STOP(transcode);
        }

        cleanup_cuda_kernels();

        return ret;
}
