#include <cinttypes>
#include <cstddef>
#include <cuda_runtime.h>

void convert_16_8_cuda(uint16_t *in, uint8_t *out, size_t in_len, cudaStream_t stream);
