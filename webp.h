#ifndef WEBP_H_C518F26F_AE0F_4765_AEB5_4F8C9EA3EDD9
#define WEBP_H_C518F26F_AE0F_4765_AEB5_4F8C9EA3EDD9

#ifdef __cplusplus
extern "C" {
#endif

enum {
        MAX_WEBP_DIMENSION = (1 << 14) - 1,
};

struct webp_encoder;
struct dec_image;
struct webp_encoder *webp_encoder_create(int quality);
unsigned long encode_webp(struct webp_encoder *enc, const struct dec_image *img,
                          unsigned long width_padding, const char *ofname,
                          const struct dec_image *orig_img);
void webp_encoder_destroy(struct webp_encoder *enc);

#ifdef __cplusplus
}
#endif

#endif // not defined WEBP_H_C518F26F_AE0F_4765_AEB5_4F8C9EA3EDD9
