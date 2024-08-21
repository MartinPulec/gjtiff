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
#include "libnvtiff.h"
#include "libtiff.hpp"
#include "utils.hpp"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

struct state_gjtiff {
        state_gjtiff(int log_level, bool use_libtiff);
        ~state_gjtiff();
        int log_level;
        bool use_libtiff; // if nvCOMP not found, enforce libtiff
        cudaStream_t stream;
        struct nvtiff_state *state_nvtiff;
        struct libtiff_state *state_libtiff;
        // GPUJPEG
        struct gpujpeg_encoder *gj_enc{};
};

state_gjtiff::state_gjtiff(int l, bool u)
    : log_level(l), use_libtiff(u), state_libtiff(libtiff_create(l))
{
        CHECK_CUDA(cudaStreamCreate(&stream));
        state_nvtiff = nvtiff_init(stream, l);
        assert(state_nvtiff != nullptr);
        gj_enc = gpujpeg_encoder_create(stream);
        assert(gj_enc != nullptr);
}

state_gjtiff::~state_gjtiff()
{
        gpujpeg_encoder_destroy(gj_enc);
        nvtiff_destroy(state_nvtiff);
        CHECK_CUDA(cudaStreamDestroy(stream));
        libtiff_destroy(state_libtiff);
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
        printf("Decoding from file %s... \n", fname);
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
        return libtiff_decode(s->state_libtiff, fname, s->stream);
}

static void encode_jpeg(int log_level, struct state_gjtiff *s,
                        struct dec_image uncomp, const char *ofname)
{
        gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);

        gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = uncomp.width;
        param_image.height = uncomp.height;
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 25, 0)
        param_image.comp_count = comp_count;
#endif
        param_image.color_space =
            uncomp.comp_count == 1 ? GPUJPEG_YCBCR_JPEG : GPUJPEG_RGB;
        param_image.pixel_format =
            uncomp.comp_count == 1 ? GPUJPEG_U8 : GPUJPEG_444_U8_P012;
        gpujpeg_encoder_input encoder_input;
        gpujpeg_encoder_input_set_gpu_image(&encoder_input, uncomp.data);
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
        printf("\t-v[v]    - be verbose (2x for more messages)\n");
        printf("\n");
        printf("Output filename will be \"basename ${name%%.*}.jpg\"\n");
        printf("Output directory must exist, implicitly \".\"\n\n");
        printf("If the '-' is given as an argument, newline-separated list of "
               "file "
               "names\nis read from stdin.\n");
}

/// @returns filename to process either from argv or read from stdin
static char *get_next_ifname(bool from_stdin, char ***argv, char *buf,
                             size_t buflen)
{
        if (!from_stdin) {
                return *(*argv)++;
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
        return buf;
}

int main(int argc, char **argv)
{
        int log_level = 0;
        bool use_libtiff = false;
        char ofname[1024] = "./";

        int opt = 0;
        while ((opt = getopt(argc, argv, "+dhlo:v")) != -1) {
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
                        if (strlen(argv[-1]) > 2) { // -o<dir>
                                snprintf(ofname, sizeof ofname, "%s/",
                                         argv[-1] + 2);
                        } else { // -o <dir>
                                assert(argv[0] != nullptr);
                                snprintf(ofname, sizeof ofname, "%s/", argv[0]);
                                argv++;
                        }
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

        struct state_gjtiff state(log_level, use_libtiff);
        int ret = EXIT_SUCCESS;

        char path_buf[PATH_MAX];
        const bool fname_from_stdin = strcmp(argv[0], "-") == 0;
        const size_t d_pref_len = strlen(ofname);
        argv += optind;
        while (char *ifname = get_next_ifname(fname_from_stdin, &argv, path_buf,
                                              sizeof path_buf)) {
                TIMER_START(transcode, log_level);
                set_ofname(ifname, ofname + d_pref_len,
                           sizeof ofname - d_pref_len);

                struct dec_image dec = decode_tiff(&state, ifname);
                if (dec.data == nullptr) {
                        ret = ERR_SOME_FILES_NOT_TRANSCODED;
                        continue;
                }
                encode_jpeg(log_level, &state, dec, ofname);
                TIMER_STOP(transcode, log_level);
        }

        return ret;
}
