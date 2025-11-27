#include "nd_processing.cuh"
#include "utils.h"

#include <stdlib.h>

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream)
{
        ERROR_MSG("Normalized differential not imlemented!\n");
        abort();
}
