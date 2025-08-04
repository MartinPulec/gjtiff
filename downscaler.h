#ifndef DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9
#define DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

#ifdef __cplusplus
extern "C" {
#endif

struct downscaler_state;
struct owned_image;

struct downscaler_state *downscaler_init(cudaStream_t stream);
struct dec_image downscale(struct downscaler_state *state, int downscale_factor,
                           const struct dec_image *in);
void downscaler_destroy(struct downscaler_state *s);

struct owned_image *scale(struct downscaler_state *state, int new_width,
                          int new_height, const struct dec_image *src);
struct owned_image *scale_pitch(struct downscaler_state *state, int new_width,
                                int x, size_t xpitch, int new_height, int y,
                                size_t dst_lines, const struct dec_image *src);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9
