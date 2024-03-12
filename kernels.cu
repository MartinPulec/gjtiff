#include "kernels.hpp"

__global__ void kernel_convert_16_8(uint16_t *in, uint8_t *out, size_t datalen) {
  int position = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
  if (position > datalen) {
    return;
  }
  out[position] = in[position] / 256;
}

uint8_t *convert16_8(uint16_t *in, size_t in_len, cudaStream_t stream) {
  uint8_t *out = nullptr;
  size_t count = in_len / 2;
  cudaMalloc(&out, count);
  kernel_convert_16_8<<<dim3((in_len+255)/256), dim3(256), 0, stream>>>(in, out, count);
  return out;
}
