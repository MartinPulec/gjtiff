#include "webp.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webp/encode.h>

#include "defs.h" // dec_image
#include "utils.h"

enum {
        DEFAULT_WEBP_QUALITY = 75, ///< equals GJ default
};

struct webp_encoder {
        int quality;
        unsigned char *chroma;
        size_t chroma_allocated;
};

static int my_write(const uint8_t *data, size_t data_size,
                    const WebPPicture *picture)
{
        FILE *outfile = picture->custom_ptr;
        return fwrite(data, data_size, 1, outfile) == 1;
}

struct webp_encoder *webp_encoder_create(int quality)
{
        struct webp_encoder *enc = calloc(1, sizeof *enc);
        enc->quality = quality == -1 ? DEFAULT_WEBP_QUALITY : quality;
        return enc;
}

#define STATUS_TO_NAME(x) {x, #x}
static const struct {
        enum WebPEncodingError rc;
        const char *str;
} webp_err_map[] = {
    STATUS_TO_NAME(VP8_ENC_OK),
    STATUS_TO_NAME(VP8_ENC_ERROR_OUT_OF_MEMORY),
    STATUS_TO_NAME(VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY),
    STATUS_TO_NAME(VP8_ENC_ERROR_NULL_PARAMETER),
    STATUS_TO_NAME(VP8_ENC_ERROR_INVALID_CONFIGURATION),
    STATUS_TO_NAME(VP8_ENC_ERROR_BAD_DIMENSION),
    STATUS_TO_NAME(VP8_ENC_ERROR_PARTITION0_OVERFLOW),
    STATUS_TO_NAME(VP8_ENC_ERROR_PARTITION_OVERFLOW),
    STATUS_TO_NAME(VP8_ENC_ERROR_BAD_WRITE),
    STATUS_TO_NAME(VP8_ENC_ERROR_FILE_TOO_BIG),
    STATUS_TO_NAME(VP8_ENC_ERROR_USER_ABORT),
};

const char *err_to_name(enum WebPEncodingError rc)
{
        for (unsigned i = 0; i < ARR_SIZE(webp_err_map); ++i) {
                if (webp_err_map[i].rc == rc) {
                        return webp_err_map[i].str;
                }
        }
        return "(unknown)";
}

unsigned long encode_webp(struct webp_encoder *enc, const struct dec_image *img,
                          unsigned long width_padding, const char *ofname,
                          const struct dec_image *orig_img)
{
        struct WebPConfig webp_config;

        int ok = WebPConfigInit(&webp_config);
        if (ok) {
                webp_config.quality = enc->quality;
                webp_config.method = 2;   // 0 = fast (fails for bigger imgs), 4 - default
                webp_config.segments = 1; // 4 - max
                webp_config.partitions = 3;
                webp_config.thread_level = 1;
                ok = WebPValidateConfig(&webp_config);
        }

        if (!ok) {
                ERROR_MSG( "[webp] not OK!\n");
                return 0;
        }

        unsigned long len = (unsigned long)img->width * img->height *
                            img->comp_count;

        const size_t req_chroma_len = (((size_t)img->width + 1) / 2) *
                                      (((size_t)img->height + 1) / 2);
        if (req_chroma_len > enc->chroma_allocated) {
                enc->chroma = realloc(enc->chroma, req_chroma_len);
                memset(enc->chroma + enc->chroma_allocated, 128,
                       req_chroma_len - enc->chroma_allocated);
                enc->chroma_allocated = req_chroma_len;
        }

        struct WebPPicture webp_picture;
        ok = WebPPictureInit(&webp_picture);
        assert(ok);
        webp_picture.writer = my_write;
        webp_picture.custom_ptr = enc;

        webp_picture.use_argb = 0;
        webp_picture.colorspace = img->alpha != NULL ? WEBP_YUV420A
                                                     : WEBP_YUV420;
        webp_picture.width = img->width;
        webp_picture.height = img->height;

        if (img->comp_count == 1) {
                webp_picture.y = img->data;
                webp_picture.a = img->alpha;
                webp_picture.y_stride = img->width + width_padding;
                webp_picture.uv_stride = (img->width + 1) / 2;
                webp_picture.u = enc->chroma;
                webp_picture.v = enc->chroma;
        } else {
                webp_picture.y = orig_img->data;
                webp_picture.a = orig_img->alpha;
                webp_picture.y_stride = orig_img->width;
                webp_picture.uv_stride = (orig_img->width + 1) / 2;
                webp_picture.u = orig_img->data +
                                      webp_picture.y_stride *
                                          orig_img->height;
                webp_picture.v = webp_picture.u +
                                      webp_picture.uv_stride *
                                          ((orig_img->height + 1) / 2);

                ptrdiff_t diff = img->data - orig_img->data;
                diff /= 3; // not rgb
                int x = diff % orig_img->width;
                int y = diff / orig_img->width;
                webp_picture.y += x + (y * webp_picture.y_stride);
                webp_picture.u += x / 2 + (y / 2 * webp_picture.uv_stride);
                webp_picture.v += x / 2 + (y / 2 * webp_picture.uv_stride);
                webp_picture.a += x + (y * webp_picture.y_stride);
        }
        webp_picture.a_stride = webp_picture.y_stride;

        FILE *outfile = fopen(ofname, "wb");
        if (outfile == NULL) {
                ERROR_MSG( "[webp] cannot create %s!\n", ofname);
                return 0;
        }
        webp_picture.custom_ptr = outfile;

retry:
        WebPEncode(&webp_config, &webp_picture);
        if (webp_picture.error_code == VP8_ENC_ERROR_PARTITION0_OVERFLOW &&
            webp_config.method < 4) {
                webp_config.method += 1;
                goto retry;
        }
        if (webp_picture.error_code != VP8_ENC_OK) {
                ERROR_MSG("[webp] Encode failed: %s (%d)!\n",
                          err_to_name(webp_picture.error_code),
                          (int)webp_picture.error_code);
                len = 0;
        }
        fclose(outfile);

        return len;
}
void webp_encoder_destroy(struct webp_encoder *enc)
{
        if (enc == NULL) {
                return;
        }
        free(enc);
}

