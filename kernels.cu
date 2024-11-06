#include "kernels.hpp"

#include <cassert>
#include <cstdio>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_statistics_functions.h>
#include <type_traits>

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
  if (position >= count) {
    return;
  }

        float normalized = __saturatef(in[position] / scale);
#ifdef GAMMA
        normalized = pow(normalized, GAMMA);
#endif
        out[position] = normalized * 255;
}

static struct {
        struct {
                void *data;
                int len;
                void *d_res;
        } stat[2];
} state;
/// indices to state.scratch
enum {
        STDDEV_MEAN = 0,
        MAX = 1,
};

template <typename T> struct second_param;
template <typename Ret, typename T1, typename T2, typename... Args>
struct second_param<Ret(T1, T2, Args...)> {
        using type = T2;
};
// Helper to deduce the function type
template <typename Func>
using size_param_t = std::remove_pointer_t<typename second_param<Func>::type>;

void convert_16_8_normalize_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream)
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

        // int in NPP 12.3 while size_t in 12.6
        size_param_t<decltype(nppiMeanStdDevGetBufferHostSize_16u_C1R)>
            stddev_scratch_len_req = 0;
        size_param_t<decltype(nppiMeanStdDevGetBufferHostSize_16u_C1R)>
            max_scratch_len_req = 0;

        // GetBufferHostSize_16s_C1R_Ctx(ROI, &BufferSize, NppStreamContext);
        nppiMeanStdDevGetBufferHostSize_16u_C1R(ROI, &stddev_scratch_len_req);
        if ((int)stddev_scratch_len_req > state.stat[STDDEV_MEAN].len) {
                cudaMallocHost((void **)(&state.stat[STDDEV_MEAN].data),
                               stddev_scratch_len_req);
                state.stat[STDDEV_MEAN].len = (int)stddev_scratch_len_req;
        }
        nppiMaxGetBufferHostSize_16u_C1R(ROI, &max_scratch_len_req);
        if ((int)max_scratch_len_req > state.stat[MAX].len) {
                cudaMallocHost((void **)(&state.stat[MAX].data),
                               max_scratch_len_req);
                state.stat[MAX].len = (int)max_scratch_len_req;
        }
        // printf("%d\n", BufferSize);
        if (state.stat[STDDEV_MEAN].d_res == nullptr) {
                cudaMalloc((void **)(&state.stat[STDDEV_MEAN].d_res),
                           2 * sizeof(Npp64f));
        }
        if (state.stat[MAX].d_res == nullptr) {
                cudaMalloc((void **)(&state.stat[MAX].d_res), sizeof(Npp16u));
        }
        NppStatus rc = NPP_NO_ERROR;
        rc = nppiMean_StdDev_16u_C1R(
            (Npp16u *)in->data, ROI.width * 2, ROI,
            (Npp8u *)state.stat[STDDEV_MEAN].data,
            &((Npp64f *)state.stat[STDDEV_MEAN].d_res)[0],
            &((Npp64f *)state.stat[STDDEV_MEAN].d_res)[1]);
        assert(rc == 0);
        Npp64f stddev_mean_res[2];
        cudaMemcpyAsync(stddev_mean_res, state.stat[STDDEV_MEAN].d_res, sizeof stddev_mean_res, cudaMemcpyDeviceToHost, stream);

        rc = nppiMax_16u_C1R((Npp16u *)in->data, ROI.width * 2, ROI,
                                     (Npp8u*)state.stat[MAX].data, (Npp16u*)state.stat[MAX].d_res);
        assert(rc == 0);
        Npp16u max_res = 0;
        cudaMemcpyAsync(&max_res, state.stat[MAX].d_res, sizeof max_res, cudaMemcpyDeviceToHost, stream);

        if (log_level >= 1) {
                printf("MEAN: %f STDDEV: %f MAX: %hu\n", stddev_mean_res[0],
                       stddev_mean_res[1], max_res);
        }

        const size_t count = (size_t)in->width * in->height;
        // scale to 0..\mu+2*\sigma
        float scale = MIN(stddev_mean_res[0] + 2 * stddev_mean_res[1], max_res);
        kernel_convert_16_8<<<dim3((count + 255) / 256), dim3(256), 0,
                              stream>>>((uint16_t *)in->data, out, count,
                                        scale);
}

/*                             _                 __    _  __   _     
 *    ___ ___  _ __ ___  _ __ | | _____  __      \ \  / |/ /_ | |__  
 *   / __/ _ \| '_ ` _ \| '_ \| |/ _ \ \/ /  _____\ \ | | '_ \| '_ \ 
 *  | (_| (_) | | | | | | |_) | |  __/>  <  |_____/ / | | (_) | |_) |
 *   \___\___/|_| |_| |_| .__/|_|\___/_/\_\      /_/  |_|\___/|_.__/ 
 *                      |_|                                          
*/
__global__ void kernel_convert_complex_int(const int16_t *in, uint16_t *out,
                                           size_t datalen)
{
        unsigned int position =
            threadIdx.x + (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
        if (position >= datalen) {
                return;
        }
        out[position] = sqrt(pow(in[2 * position], 2) + pow(in[2 * position + 1], 2));
}
void convert_complex_int_to_uint16(const int16_t *in, uint16_t *out,
                                  size_t count, cudaStream_t stream)
{
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
  if (position >= datalen) {
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

template<typename t>
__global__ void kernel_convert_remove_pitch(t *in, t *out,
                                            int width, int spitch)
{
        int position_x = threadIdx.x + blockIdx.x * blockDim.x;
        if (position_x >= width) {
                return;
        }
        int position_y = threadIdx.y + blockIdx.y * blockDim.y;
        out[position_y * width + position_x] =
            in[position_y * spitch + position_x];
}

/**
 * This function is not 100% necessary since GPUJPEG supports pitched
 * input (but currently just RGB) but it won't perhaps dealinkg with it since
 * CUDA kernels are quite fast
 */
void convert_remove_pitch(uint8_t *in, uint8_t *out, int width, int spitch,
                          int height, void *stream)
{
        kernel_convert_remove_pitch<uint8_t><<<dim3((width + 255) / 256, height),
                                      dim3(256), 0, (cudaStream_t)stream>>>(
            in, out, width, spitch);
}


/**
 * This function is not 100% necessary since GPUJPEG supports pitched
 * input (but currently just RGB) but it won't perhaps dealinkg with it since
 * CUDA kernels are quite fast
 */
void convert_remove_pitch_16(uint16_t *in, uint16_t *out, int width, int spitch,
                          int height, void *stream)
{
        kernel_convert_remove_pitch<uint16_t><<<dim3((width + 255) / 256, height),
                                      dim3(256), 0, (cudaStream_t)stream>>>(
            in, out, width, spitch / 2);
}

void cleanup_cuda_kernels()
{
        for (unsigned i = 0; i < ARR_SIZE(state.stat); ++i) {
                cudaFreeHost(state.stat[i].data);
                cudaFree(state.stat[i].d_res);
        }
}
