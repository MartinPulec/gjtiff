#include <cinttypes>
#include <cstddef>
#include <cuda_runtime.h>

uint8_t *convert16_8(uint16_t *in, size_t in_len, cudaStream_t stream);
