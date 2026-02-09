#ifndef WEBP_H_F0053B0C_A18F_4FEB_A39D_80ABC05A14EC
#define WEBP_H_F0053B0C_A18F_4FEB_A39D_80ABC05A14EC

#ifdef __cplusplus
extern "C" {
#endif

struct dec_image;

void png_init();
unsigned long encode_png(const struct dec_image *img,
                         unsigned long width_padding, const char *ofname);

#ifdef __cplusplus
}
#endif

#endif // defined WEBP_H_F0053B0C_A18F_4FEB_A39D_80ABC05A14EC
