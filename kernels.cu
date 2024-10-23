#include "kernels.hpp"

#include <cassert>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_statistics_functions.h>
#include <printf.h>

#include "defs.h"

#define GAMMA 2

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

        float normalized = __saturatef(in[position] / scale);
#ifdef GAMMA
        normalized = pow(normalized, GAMMA);
#endif
        out[position] = normalized * 255;
}

static struct {
        Npp64f *d_res;
        Npp8u *scratch;
        int scratch_len;

} state;

void convert_16_8_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream)
{
        if (nppGetStream() != stream) {
                nppSetStream(stream);
        }
        // NppStreamContext NppStreamContext;
        // rc = nppGetStreamContext(&NppStreamContext);
        // assert(rc == 0);
        NppiSize ROI;
        ROI.width = in->width;
        ROI.height = in->height;
        int scratch_len_req = 0;
        // GetBufferHostSize_16s_C1R_Ctx(ROI, &BufferSize, NppStreamContext);
        nppiMeanStdDevGetBufferHostSize_16u_C1R(ROI, &scratch_len_req);
        if (scratch_len_req > state.scratch_len) {
                cudaMallocHost((void **)(&state.scratch), scratch_len_req);
                state.scratch_len = scratch_len_req;
        }
        // printf("%d\n", BufferSize);
        if (state.d_res == nullptr) {
                cudaMalloc((void **)(&state.d_res), 2 * sizeof(Npp64f));
        }
        NppStatus rc = NPP_NO_ERROR;
        rc = nppiMean_StdDev_16u_C1R((Npp16u *)in->data, ROI.width * 2, ROI,
                                     state.scratch, &state.d_res[0], &state.d_res[1]);
        assert(rc == 0);
        Npp64f res[2];
        cudaMemcpyAsync(res, state.d_res, sizeof res, cudaMemcpyDeviceToHost, stream);
        if (log_level >= 1) {
                printf("MEAN: %f STDDEV: %f\n", res[0], res[1]);
        }

        const size_t count = (size_t)in->width * in->height;
        // scale to 0..\mu+2*\sigma
        kernel_convert_16_8<<<dim3((count + 255) / 256), dim3(256), 0,
                              stream>>>((uint16_t *)in->data, out, count,
                                        res[0] + 2 * res[1]);
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

void cleanup_cuda_kernels()
{
        cudaFreeHost(state.scratch);
        cudaFree(state.d_res);
}
