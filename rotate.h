#ifndef ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5
#define ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

#ifdef __cplusplus
extern "C" {
#endif

struct rotate_state;

struct rotate_state *rotate_init(cudaStream_t stream);
struct dec_image rotate(struct rotate_state *state, const struct dec_image *in);
void rotate_destroy(struct rotate_state *s);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define ROTATE_H_F705B78A_DA37_4B99_982C_D3B974ED90E5
