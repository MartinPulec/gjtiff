#include <cuda_runtime.h>

#include "defs.h"

struct nvtiff_state;

struct nvtiff_state *nvtiff_init(cudaStream_t stream, int log_level);
struct dec_image nvtiff_decode(struct nvtiff_state *state, const char *fname);
void nvtiff_destroy(struct nvtiff_state *s);
