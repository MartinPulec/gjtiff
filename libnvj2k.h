#ifndef LIBNVJ2K_H_8D836DD7_5D39_4801_A6FA_C494644A1A0
#define LIBNVJ2K_H_8D836DD7_5D39_4801_A6FA_C494644A1A0

#include <cuda_runtime.h>  // for cudaStream_t

#include "defs.h"          // for dec_image

struct nvj2k_state;

struct nvj2k_state *nvj2k_init(cudaStream_t stream);
struct dec_image nvj2k_decode(struct nvj2k_state *state, const char *fname);
void nvj2k_destroy(struct nvj2k_state *s);

#endif // defined LIBNVJ2K_H_8D836DD7_5D39_4801_A6FA_C494644A1A0
