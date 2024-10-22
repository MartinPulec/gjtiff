#include "kernels.hpp"

#include "defs.h"

/*  __     __   .______           ___        ___   .______   
 * /_ |   / /   |   _  \          \  \      / _ \  |   _  \  
 *  | |  / /_   |  |_)  |     _____\  \    | (_) | |  |_)  | 
 *  | | | '_ \  |   _  <     |______>  >    > _ <  |   _  <  
 *  | | | (_) | |  |_)  |          /  /    | (_) | |  |_)  | 
 *  |_|  \___/  |______/          /__/      \___/  |______/  
*/
__global__ void kernel_convert_16_8(uint16_t *in, uint8_t *out, size_t count, float scale) {
  int position = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
  if (position > count) {
    return;
  }
  out[position] = __saturatef(in[position] / scale) * 255;
}

void convert_16_8_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream)
{
        const size_t count = (size_t)in->width * in->height;
        kernel_convert_16_8<<<dim3((count + 255) / 256), dim3(256), 0,
                              stream>>>((uint16_t *)in->data, out, count, 343);
}



__global__ void kernel_convert_complex_int(const uint8_t *in, uint8_t *out,
                                           size_t datalen)
{
        unsigned int position =
            threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
        if (position > datalen) {
                return;
        }
        out[position] = in[4 * position + 1]; // take just MSB from real part
}
void convert_complex_int(const uint8_t *in, uint8_t *out, size_t in_len,
                         cudaStream_t stream)
{
        const size_t count = in_len / 4;
        kernel_convert_complex_int<<<dim3((count + 255) / 256), dim3(256), 0,
                                     stream>>>(in, out, count);
}


__global__ void kernel_convert_rgba_grayscale(uint8_t *in, uint8_t *out, size_t datalen) {
  int position = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
  if (position > datalen) {
    return;
  }
  out[position] = in[position * 4];
}

void convert_rgba_grayscale(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream)
{
        kernel_convert_rgba_grayscale<<<dim3((pix_count + 255) / 256),
                                        dim3(256), 0, (cudaStream_t)stream>>>(
            in, out, pix_count);
}

__global__ void kernel_convert_rgba_rgb(uint8_t *in, uint8_t *out, size_t datalen) {
  int position = threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
  if (position > datalen) {
    return;
  }
  out[position * 3] = in[position * 4];
  out[position * 3 + 1] = in[position * 4 + 1];
  out[position * 3 + 1] = in[position * 4 + 1];
}

void convert_rgba_rgb(uint8_t *in, uint8_t *out, size_t pix_count,
                            void *stream)
{
        kernel_convert_rgba_rgb<<<dim3((pix_count + 255) / 256), dim3(256), 0,
                                  (cudaStream_t)stream>>>(in, out, pix_count);
}
