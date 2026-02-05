#ifndef PROCESSING_H_1A2274A8_D9D4_4F40_BC59_2DF934146EC5
#define PROCESSING_H_1A2274A8_D9D4_4F40_BC59_2DF934146EC5

#include <cuda_runtime.h>

#include "defs.h"

struct nd_data {
        unsigned count;
        struct {
                int width;
                int height;
                void *data;
                unsigned char *alpha;
        } img[2];
};

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct nd_data *in, cudaStream_t stream);

#endif
