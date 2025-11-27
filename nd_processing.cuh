#ifndef PROCESSING_H_1A2274A8_D9D4_4F40_BC59_2DF934146EC5
#define PROCESSING_H_1A2274A8_D9D4_4F40_BC59_2DF934146EC5

#include <cuda_runtime.h>

#include "defs.h"

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream);

#endif
