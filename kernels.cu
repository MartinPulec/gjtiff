#include "kernels.h"

#include <algorithm>
#include <cstdio>
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_statistics_functions.h>
#include <type_traits>

#include "defs.h"
#include "utils.h"

#define GAMMA 2

/*                                   _ _         
 *   _ __   ___  _ __ _ __ ___   __ _| (_)_______ 
 *  | '_ \ / _ \| '__| '_ ` _ \ / _` | | |_  / _ \
 *  | | | | (_) | |  | | | | | | (_| | | |/ /  __/
 *  |_| |_|\___/|_|  |_| |_| |_|\__,_|_|_/___\___|
 */
 template <typename t>
__global__ void kernel_normalize(t *in, uint8_t *out, size_t count, float scale) {
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

#define OBTAIN_NPP_CTX                                                         \
        static thread_local NppStreamContext nppStreamCtx{};                   \
        static thread_local cudaStream_t saved_stream = (cudaStream_t)1;       \
        if (stream != saved_stream) {                                          \
                init_npp_context(&nppStreamCtx, (cudaStream_t)stream);         \
                saved_stream = (cudaStream_t)stream;                           \
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
        MEAN_STDDEV = 0,
        MAX = 1,
};

template <typename T> struct second_param;
template <typename Ret, typename T1, typename T2, typename... Args>
struct second_param<Ret(T1, T2, Args...)> {
        using type = T2;
};
// Helper to deduce the function type
template <typename Func>
using size_param_t = typename std::remove_pointer<typename second_param<Func>::type>::type;

#ifdef NPP_NEW_API
#define CONTEXT , nppStreamCtx
#else
#define CONTEXT
#endif

struct normalize_8b {
        using nv_type = uint8_t; // typedefed as Npp8 in NPP
        constexpr static auto mean_stddev_size = NPP_CONTEXTIZE(
            nppiMeanStdDevGetBufferHostSize_8u_C1R);
        constexpr static auto mean_stddev = NPP_CONTEXTIZE(
            nppiMean_StdDev_8u_C1R);
        constexpr static auto max_size = NPP_CONTEXTIZE(
            nppiMeanStdDevGetBufferHostSize_8u_C1R);
        constexpr static auto max = NPP_CONTEXTIZE(nppiMax_8u_C1R);
};

struct normalize_16b {
        using nv_type = uint16_t; // typedefed as Npp16 in NPP
        constexpr static auto mean_stddev_size = NPP_CONTEXTIZE(
            nppiMeanStdDevGetBufferHostSize_16u_C1R);
        constexpr static auto mean_stddev = NPP_CONTEXTIZE(
            nppiMean_StdDev_16u_C1R);
        constexpr static auto max_size = NPP_CONTEXTIZE(
            nppiMeanStdDevGetBufferHostSize_16u_C1R);
        constexpr static auto max = NPP_CONTEXTIZE(nppiMax_16u_C1R);
};

template <typename t>
void normalize_cuda(struct dec_image *in, uint8_t *out, cudaStream_t stream)
{
        enum {
                MEAN = 0,
                STDDEV = 1,
                MEAN_STDDEV_RES_COUNT = STDDEV + 1,
                SIGMA_COUNT = 2, // nultiple of sigma to be added to the mean to
                                 // obtain scale
        };
#ifdef NPP_NEW_API
        OBTAIN_NPP_CTX
#else
        if (nppGetStream() != stream) {
                nppSetStream(stream);
        }
#endif
        const int bps = sizeof(typename t::nv_type);
        NppiSize ROI;
        ROI.width = in->width;
        ROI.height = in->height * in->comp_count;

        // int in NPP 12.3 while size_t in 12.6
        size_param_t<typename std::remove_pointer<decltype(t::mean_stddev_size)>::type>
            stddev_scratch_len_req = 0;
        size_param_t<typename std::remove_pointer<decltype(t::max_size)>::type>
            max_scratch_len_req = 0;

        // GetBufferHostSize_16s_C1R_Ctx(ROI, &BufferSize, NppStreamContext);
        CHECK_NPP(t::mean_stddev_size(ROI, &stddev_scratch_len_req CONTEXT));
        if ((int)stddev_scratch_len_req > state.stat[MEAN_STDDEV].len) {
                CHECK_CUDA(cudaFreeHost(state.stat[MEAN_STDDEV].data));
                CHECK_CUDA(
                    cudaMallocHost((void **)(&state.stat[MEAN_STDDEV].data),
                                   stddev_scratch_len_req));
                state.stat[MEAN_STDDEV].len = (int)stddev_scratch_len_req;
        }
        CHECK_NPP(t::max_size(ROI, &max_scratch_len_req CONTEXT));
        if ((int)max_scratch_len_req > state.stat[MAX].len) {
                CHECK_CUDA(cudaFreeHost(state.stat[MAX].data));
                CHECK_CUDA(cudaMallocHost((void **)(&state.stat[MAX].data),
                                          max_scratch_len_req));
                state.stat[MAX].len = (int)max_scratch_len_req;
        }
        // printf("%d\n", BufferSize);
        if (state.stat[MEAN_STDDEV].d_res == nullptr) {
                CHECK_CUDA(cudaMalloc((void **)(&state.stat[MEAN_STDDEV].d_res),
                                      MEAN_STDDEV_RES_COUNT * sizeof(Npp64f)));
        }
        if (state.stat[MAX].d_res == nullptr) {
                CHECK_CUDA(cudaMalloc((void **)(&state.stat[MAX].d_res),
                                      sizeof(Npp16u)));
        }
        CHECK_NPP(t::mean_stddev(
            (typename t::nv_type *)in->data, ROI.width * bps, ROI,
            (Npp8u *)state.stat[MEAN_STDDEV].data,
            &((Npp64f *)state.stat[MEAN_STDDEV].d_res)[MEAN],
            &((Npp64f *)state.stat[MEAN_STDDEV].d_res)[STDDEV]
            CONTEXT));
        Npp64f stddev_mean_res[MEAN_STDDEV_RES_COUNT];
        CHECK_CUDA(cudaMemcpyAsync(
            stddev_mean_res, state.stat[MEAN_STDDEV].d_res,
            sizeof stddev_mean_res, cudaMemcpyDeviceToHost, stream));

        CHECK_NPP(t::max((typename t::nv_type *)in->data, ROI.width * bps, ROI,
                    (Npp8u *)state.stat[MAX].data,
                    (typename t::nv_type *)state.stat[MAX].d_res CONTEXT));
        typename t::nv_type max_res = 0;
        CHECK_CUDA(cudaMemcpyAsync(&max_res, state.stat[MAX].d_res,
                                   sizeof max_res, cudaMemcpyDeviceToHost,
                                   stream));

        VERBOSE_MSG("MEAN: %f STDDEV: %f MAX: %hu\n", stddev_mean_res[MEAN],
                    stddev_mean_res[STDDEV], max_res);

        const size_t count = (size_t)in->width * in->height * in->comp_count;
        // scale to 0..\mu+2*\sigma
        const float scale = MIN(stddev_mean_res[MEAN] +
                                    SIGMA_COUNT * stddev_mean_res[STDDEV],
                                max_res);
        kernel_normalize<typename t::nv_type>
            <<<dim3((count + 255) / 256), dim3(256), 0, stream>>>(
                (typename t::nv_type *)in->data, out, count, scale);
        CHECK_CUDA(cudaGetLastError());
}

void convert_16_8_normalize_cuda(struct dec_image *in, uint8_t *out,
                                 cudaStream_t stream)
{
        normalize_cuda<normalize_16b>(in, out, stream);
}

void normalize_8(struct dec_image *in, uint8_t *out, cudaStream_t stream)
{
        normalize_cuda<normalize_8b>(in, out, stream);
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
        CHECK_CUDA(cudaGetLastError());
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
        CHECK_CUDA(cudaGetLastError());
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
        CHECK_CUDA(cudaGetLastError());
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
        CHECK_CUDA(cudaGetLastError());
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
        CHECK_CUDA(cudaGetLastError());
}

template<int comp_count>
__global__ void kernel_downscale(const uint8_t *in, uint8_t *out,
                                            int src_width, int factor)
{
        int dst_width = src_width / factor;
        int position_x = threadIdx.x + blockIdx.x * blockDim.x;
        if (position_x >= dst_width) {
                return;
        }
        int position_y = threadIdx.y + blockIdx.y * blockDim.y;
        for (int i = 0; i < comp_count; ++i) {
                out[comp_count * (position_y * dst_width + position_x) + i] =
                    in[comp_count * factor *
                           (position_y * src_width + position_x) +
                       i];
        }
}

void downscale_image_cuda(const uint8_t *in, uint8_t *out, int comp_count,
                          int src_width, int src_height, int factor,
                          void *stream)
{
        int dst_width = src_width / factor;
        int dst_height = src_height / factor;

        dim3 threads_per_block(256);
        dim3 blocks((dst_width + 255) / 256, dst_height);
        switch (comp_count) {
                case 1:
                        kernel_downscale<1><<<blocks, threads_per_block, 0,
                                           (cudaStream_t)stream>>>(
                            in, out, src_width, factor);
                        break;
                case 3:
                        kernel_downscale<3><<<blocks, threads_per_block, 0,
                                           (cudaStream_t)stream>>>(
                            in, out, src_width, factor);
                        break;
                default:
                        ERROR_MSG(
                            "Downscaling for %d channels not supported!\n",
                            comp_count);
                }
        CHECK_CUDA(cudaGetLastError());
}

void convert_u8_to_float(const uint8_t *in, float *out, size_t count, void *stream) {
        OBTAIN_NPP_CTX
        NppiSize ROI;
        ROI.width = count;
        ROI.height = 1;
        CHECK_NPP(nppiConvert_8u32f_C1R_Ctx(
            in, count, out, count * sizeof(float), ROI, nppStreamCtx));
}

void npp_float_operation(const float *in1, const float *in2,
                         float *out, char op, size_t count, void *stream) {
        OBTAIN_NPP_CTX
        NppiSize ROI;
        ROI.width = count;
        ROI.height = 1;
        int step = count * sizeof(float);
        if (op == '+') {
                nppiAdd_32f_C1R_Ctx(in1, step, in2, step, out, step, ROI, nppStreamCtx);
        } else {
                ERROR_MSG("Unexpected operation %c!\n", op);
                abort();
        }
        
}

uint32_t *get_lut(int ramp_sz, const struct ramp *ramp, void *vstream) {
        uint32_t lut[1000];
        int first_idx = ramp->src * 1000;
        for (int i = 0; i < first_idx; ++i) {
                lut[i] = ramp->dst;
        }

        for (int i = 1; i < ramp_sz - 1; ++i) {
                int a1 = ramp[-1].dst >> 24;
                int b1 = (ramp[-1].dst >> 16) & 0xFF;
                int g1 = (ramp[-1].dst >> 8) & 0xFF;
                int r1 = (ramp[-1].dst >> 0) & 0xFF;

                int a2 = ramp[0].dst >> 24;
                int b2 = (ramp[0].dst >> 16) & 0xFF;
                int g2 = (ramp[0].dst >> 8) & 0xFF;
                int r2 = (ramp[0].dst >> 0) & 0xFF;

                int first_idx = ramp[-1].src * 1000;
                int last_idx = ramp[0].src * 1000;
                double scale = last_idx - first_idx;
                for (int i = first_idx; i < last_idx; ++i) {
                         int a = a1 + i * scale * (a2 - a1) / 1000;
                         int b = b1 + i * scale * (b2 - b1) / 1000;
                         int g = g1 + i * scale * (g2 - g1) / 1000;
                         int r = r1 + i * scale * (r2 - r1) / 1000;

                         lut[i] = std::clamp(a, 0, 255) << 24 |
                                  std::clamp(b, 0, 255) << 24 |
                                  std::clamp(g, 0, 255) << 24 |
                                  std::clamp(r, 0, 255) << 24;
                }
        }
        int last_idx = ramp[ramp_sz - 1].src * 1000;
        int last_rgb = ramp[ramp_sz - 1].dst * 1000;
        for (int i = last_idx; i < 1000; ++i) {
                lut[i] = last_rgb;
        }

        uint32_t *ret = nullptr;
        auto *stream = (cudaStream_t) vstream;
        CHECK_CUDA(cudaMallocAsync(&ret, sizeof lut, stream));
        CHECK_CUDA(
            cudaMemcpyAsync(ret, lut, sizeof lut, cudaMemcpyDefault, stream));
        return ret;
}

static __global__ void kernel_convert_float_to_u32(const float *in,
                                                   uint32_t *out,
                                                   const uint32_t *lut,
                                                   size_t datalen)
{
        int position = threadIdx.x +
                       (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
        if (position > datalen) {
                return;
        }
        float val = in[position]; // range -1.0-1.0
        val += 1.0;
        val /= 2.0;
        int index= val * 1000;
        if (index < 0) {
                index = 0;
        }
        if (index > 999) {
                index = 999;
        }
        out[position] = lut[index];
}

void convert_float_to_u32(const float *d_in, uint32_t *d_out, size_t count,
                          const uint32_t *d_lut, void *stream)
{
        kernel_convert_float_to_u32<<<dim3((count+ 255) / 256),
                                        dim3(256), 0, (cudaStream_t)stream>>>(
            d_in, d_out, d_lut, count);
        CHECK_CUDA(cudaGetLastError());
}

void cleanup_cuda_kernels()
{
        for (unsigned i = 0; i < ARR_SIZE(state.stat); ++i) {
                CHECK_CUDA(cudaFreeHost(state.stat[i].data));
                CHECK_CUDA(cudaFree(state.stat[i].d_res));
        }
}
