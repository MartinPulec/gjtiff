#ifndef ROTATE_UTM_H_7575f648_a727_4b80_8752_08cc7a863020
#define ROTATE_UTM_H_7575f648_a727_4b80_8752_08cc7a863020

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

#ifdef __cplusplus
extern "C" {
#endif

struct rotate_utm_state;

struct rotate_utm_state *rotate_utm_init(cudaStream_t stream);
struct owned_image *rotate_utm(struct rotate_utm_state *state, const struct dec_image *in);
void rotate_utm_destroy(struct rotate_utm_state *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // not defined ROTATE_UTM_H_7575f648_a727_4b80_8752_08cc7a863020
