#ifndef DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9
#define DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

#ifdef __cplusplus
extern "C" {
#endif

struct downscaler_state;

struct downscaler_state *downscaler_init(cudaStream_t stream);
struct dec_image downscale(struct downscaler_state *state, int downscale_factor,
                           const struct dec_image *in);
void downscaler_destroy(struct downscaler_state *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define DOWNSCALER_H_2B01111E_0C41_482C_85FF_30D914C93B9
