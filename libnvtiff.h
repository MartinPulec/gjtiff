#ifndef LIBNVTIFF_H_D36D3982_4B39_4468_957A_D49A8ACA1A05
#define LIBNVTIFF_H_D36D3982_4B39_4468_957A_D49A8ACA1A05

#include <cuda_runtime.h>

#include "defs.h"

struct nvtiff_state;

struct nvtiff_state *nvtiff_init(cudaStream_t stream, int log_level);
struct dec_image nvtiff_decode(struct nvtiff_state *state, const char *fname);
void nvtiff_destroy(struct nvtiff_state *s);

#endif // !defined LIBNVTIFF_H_D36D3982_4B39_4468_957A_D49A8ACA1A05
