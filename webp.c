#include "webp.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webp/encode.h>

#include "defs.h" // dec_image
#include "utils.h"

struct webp_encoder {
        cudaStream_t cuda_stream;
        struct WebPConfig webp_config;
        struct WebPPicture webp_picture;

        unsigned char *chroma;
        size_t chroma_allocated;

        FILE *outfile;
};

static int my_write(const uint8_t *data, size_t data_size,
                    const WebPPicture *picture)
{
        struct webp_encoder *enc= picture->custom_ptr;
        return fwrite(data, data_size, 1, enc->outfile) == 1;
}

struct webp_encoder *webp_encoder_create()
{
        int ok = 0;
        struct webp_encoder *enc = calloc(1, sizeof *enc);

        ok = WebPConfigInit(&enc->webp_config);
        if (ok) {
                enc->webp_config.quality = 50;
                enc->webp_config.method = 2;   // 0 = fast (fails for bigger imgs), 4 - default
                enc->webp_config.segments = 1; // 4 - max
                enc->webp_config.partitions = 3;
                enc->webp_config.thread_level = 1;
                ok = WebPValidateConfig(&enc->webp_config);
        }

        if (ok) {
                ok = WebPPictureInit(&enc->webp_picture);
                enc->webp_picture.writer = my_write;
                enc->webp_picture.custom_ptr = enc;
        }

        if (!ok) {
                ERROR_MSG( "[webp] not OK!\n");
                return NULL;
        }
        return enc;
}

unsigned long encode_webp(struct webp_encoder *enc, const struct dec_image *img,
                          unsigned long width_padding, const char *ofname)
{
        if (img->comp_count != 1) {
                ERROR_MSG("Only 1 channel is supported as for now!\n");
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

        enc->webp_picture.use_argb  = 0;
        enc->webp_picture.colorspace = WEBP_YUV420;
        enc->webp_picture.y = img->data;
        enc->webp_picture.y_stride = img->width + width_padding;
        enc->webp_picture.uv_stride = (img->width + 1) / 2;
        enc->webp_picture.u = enc->chroma;
        enc->webp_picture.v = enc->chroma;
        enc->webp_picture.width = img->width;
        enc->webp_picture.height = img->height;

        enc->outfile = fopen(ofname, "wb");

retry:
        WebPEncode(&enc->webp_config, &enc->webp_picture);
        if (enc->webp_picture.error_code == VP8_ENC_ERROR_PARTITION0_OVERFLOW &&
            enc->webp_config.method < 4) {
                enc->webp_config.method += 1;
                goto retry;
        }
        if (enc->webp_picture.error_code != VP8_ENC_OK) {
                ERROR_MSG("[webp] Encode failed: %d!\n",
                          enc->webp_picture.error_code);
                len = 0;
        }
        fclose(enc->outfile);

        return len;
}
void webp_encoder_destroy(struct webp_encoder *enc)
{
        if (enc == NULL) {
                return;
        }
        free(enc);
}

