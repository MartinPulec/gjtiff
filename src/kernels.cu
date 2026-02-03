#include "kernels.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>                       // for atoi,getenv
#include <nppcore.h>
#include <nppdefs.h>
#include <nppi_statistics_functions.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <type_traits>

#include "cuda_common.cuh"
#include "defs.h"
#include "utils.h"

// defined in MTD_MSIL2A.xml for B0-B12
constexpr float BOA_ADD_OFFSET = -1000;
// https://docs.sentinel-hub.com/api/latest/data/sentinel-2-l1c/ and also in the
// above XML
constexpr float BOA_QUANTIFICATION_VALUE = 10000;

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

enum {
        MEAN_STDDEV,
        MIN_MAX,
        NB_STATS,
};

static struct {
        struct {
                void *data;  // scratchpad
                int len;     // scartchpad size
                void *d_res;
        } stat[NB_STATS];

        uint8_t *d_yuv420;
        size_t d_yuv420_allocated;
} state;

template <typename T> struct second_param;
template <typename Ret, typename T1, typename T2, typename... Args>
struct second_param<Ret(T1, T2, Args...)> {
        using type = T2;
};
// Helper to deduce the function type
template <typename Func>
using size_param_t = typename std::remove_pointer<typename second_param<Func>::type>::type;

struct normalize_8b {
        using nv_type = uint8_t; // typedefed as Npp8 in NPP
        constexpr static auto mean_stddev_size =
            nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx;
        constexpr static auto mean_stddev = nppiMean_StdDev_8u_C1R_Ctx;
        constexpr static auto min_max_size = nppiMinMaxGetBufferHostSize_8u_C1R_Ctx;
        constexpr static auto min_max = nppiMinMax_8u_C1R_Ctx;
};

struct normalize_16b {
        using nv_type = uint16_t; // typedefed as Npp16 in NPP
        constexpr static auto mean_stddev_size =
            nppiMeanStdDevGetBufferHostSize_16u_C1R_Ctx;
        constexpr static auto mean_stddev = nppiMean_StdDev_16u_C1R_Ctx;
        constexpr static auto min_max_size = nppiMinMaxGetBufferHostSize_16u_C1R_Ctx;
        constexpr static auto min_max = nppiMinMax_16u_C1R_Ctx;
};

template <typename t>
static void normalize_cuda(struct dec_image *in, uint8_t *out,
                            cudaStream_t stream)
{
        enum {
                MEAN_IDX = 0,
                MIN_IDX = 0,
                STDDEV_IDX = 1,
                MAX_IDX = 1,
                RES_COUNT = STDDEV_IDX + 1, // both minmax and meanstddev
                SIGMA_COUNT = 2, // nultiple of sigma to be added to the mean to
                                 // obtain scale
        };
#ifdef NPP_NEW_API
        static thread_local NppStreamContext nppStreamCtx{};
        static thread_local cudaStream_t saved_stream = (cudaStream_t) 1;
        if (stream != saved_stream) {
                init_npp_context(&nppStreamCtx, stream);
                saved_stream = stream;
        }
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
        size_param_t<typename std::remove_pointer<decltype(t::min_max_size)>::type>
            min_max_scratch_len_req = 0;

        // GetBufferHostSize_16s_C1R_Ctx(ROI, &BufferSize, NppStreamContext);
        CHECK_NPP(
            t::mean_stddev_size(ROI, &stddev_scratch_len_req, nppStreamCtx));
        if ((int)stddev_scratch_len_req > state.stat[MEAN_STDDEV].len) {
                CHECK_CUDA(cudaFreeHost(state.stat[MEAN_STDDEV].data));
                CHECK_CUDA(
                    cudaMallocHost((void **)(&state.stat[MEAN_STDDEV].data),
                                   stddev_scratch_len_req));
                state.stat[MEAN_STDDEV].len = (int)stddev_scratch_len_req;
        }
        CHECK_NPP(t::min_max_size(ROI, &min_max_scratch_len_req, nppStreamCtx));
        if ((int)min_max_scratch_len_req > state.stat[MIN_MAX].len) {
                CHECK_CUDA(cudaFreeHost(state.stat[MIN_MAX].data));
                CHECK_CUDA(cudaMallocHost((void **)(&state.stat[MIN_MAX].data),
                                          min_max_scratch_len_req));
                state.stat[MIN_MAX].len = (int)min_max_scratch_len_req;
        }
        // printf("%d\n", BufferSize);
        if (state.stat[MEAN_STDDEV].d_res == nullptr) {
                CHECK_CUDA(cudaMalloc((void **)(&state.stat[MEAN_STDDEV].d_res),
                                      RES_COUNT * sizeof(Npp64f)));
        }
        if (state.stat[MIN_MAX].d_res == nullptr) {
                CHECK_CUDA(cudaMalloc((void **)(&state.stat[MIN_MAX].d_res),
                                      RES_COUNT * sizeof(Npp16u)));
        }
        CHECK_NPP(t::mean_stddev(
            (typename t::nv_type *)in->data, ROI.width * bps, ROI,
            (Npp8u *)state.stat[MEAN_STDDEV].data,
            &((Npp64f *)state.stat[MEAN_STDDEV].d_res)[MEAN_IDX],
            &((Npp64f *)state.stat[MEAN_STDDEV].d_res)[STDDEV_IDX], nppStreamCtx));
        Npp64f stddev_mean_res[RES_COUNT];
        CHECK_CUDA(cudaMemcpyAsync(
            stddev_mean_res, state.stat[MEAN_STDDEV].d_res,
            sizeof stddev_mean_res, cudaMemcpyDeviceToHost, stream));

        CHECK_NPP(t::min_max(
            (typename t::nv_type *)in->data, ROI.width * bps, ROI,
            &((typename t::nv_type *)state.stat[MIN_MAX].d_res)[MIN_IDX],
            &((typename t::nv_type *)state.stat[MIN_MAX].d_res)[MAX_IDX],
            (Npp8u *)state.stat[MIN_MAX].data, nppStreamCtx));
        typename t::nv_type min_max_res[RES_COUNT] = {};
        CHECK_CUDA(cudaMemcpyAsync(min_max_res, state.stat[MIN_MAX].d_res,
                                   sizeof min_max_res,
                                   cudaMemcpyDeviceToHost, stream));

        VERBOSE_MSG("MEAN: %f STDDEV: %f MIN: %hu MAX: %hu\n",
                    stddev_mean_res[MEAN_IDX], stddev_mean_res[STDDEV_IDX],
                    min_max_res[MIN_IDX], min_max_res[MAX_IDX]);

        const size_t count = (size_t)in->width * in->height * in->comp_count;
        // scale to 0..\mu+2*\sigma
        float scale = MIN(stddev_mean_res[MEAN_IDX] +
                              SIGMA_COUNT * stddev_mean_res[STDDEV_IDX],
                          min_max_res[MAX_IDX]);
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

template<bool want_alpha>
static __global__ void kernel_combine(struct dec_image out,
                                      struct dec_image in1,
                                      struct dec_image in2,
                                      struct dec_image in3)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width|| out_y >= out.height) {
                return;
        }

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        float abs_pos_src_x = rel_pos_src_x * in1.width;
        float abs_pos_src_y = rel_pos_src_y * in1.height;
        out.data[3 * (out_x + out_y * out.width)] = bilinearSample(
            in1.data, in1.width, 1, in1.height, abs_pos_src_x, abs_pos_src_y);
        if (want_alpha) {
                out.alpha[out_x + (out_y * out.width)] = bilinearSample(
                    in1.alpha, in1.width, 1, in1.height, abs_pos_src_x,
                    abs_pos_src_y);
        }
        abs_pos_src_x = rel_pos_src_x * in2.width;
        abs_pos_src_y = rel_pos_src_y * in2.height;
        out.data[3 * (out_x + out_y * out.width) + 1] = bilinearSample(
            in2.data, in2.width, 1, in2.height, abs_pos_src_x, abs_pos_src_y);
        abs_pos_src_x = rel_pos_src_x * in3.width;
        abs_pos_src_y = rel_pos_src_y * in3.height;
        out.data[3 * (out_x + out_y * out.width) + 2] = bilinearSample(
            in3.data, in3.width, 1, in3.height, abs_pos_src_x, abs_pos_src_y);
}

void combine_images_cuda(struct dec_image *out, const struct dec_image *in1,
                    const struct dec_image *in2, const struct dec_image *in3,
                    cudaStream_t stream)
{
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        if (alpha_wanted) {
                kernel_combine<true>
                    <<<grid, block, 0, stream>>>(*out, *in1, *in2, *in3);
        } else {
                kernel_combine<false>
                    <<<grid, block, 0, stream>>>(*out, *in1, *in2, *in3);
        }
        CHECK_CUDA(cudaGetLastError());
}

enum {
        YUV_FIX = 16,
        YUV_HALF = 1 << (YUV_FIX - 1),
};
static __device__ int VP8ClipUV(int uv, int rounding)
{
        uv = (uv + rounding + (128 << (YUV_FIX + 2))) >> (YUV_FIX + 2);
        return ((uv & ~0xff) == 0) ? uv : (uv < 0) ? 0 : 255;
}

static __device__ int VP8RGBToY(int r, int g, int b, int rounding)
{
        const int luma = 16839 * r + 33059 * g + 6420 * b;
        return (luma + rounding + (16 << YUV_FIX)) >> YUV_FIX; // no need to
                                                               // clip
}

static __device__ int VP8RGBToU(int r, int g, int b, int rounding)
{
        const int u = -9719 * r - 19081 * g + 28800 * b;
        return VP8ClipUV(u, rounding);
}

static __device__ int VP8RGBToV(int r, int g, int b, int rounding)
{
        const int v = +28800 * r - 24116 * g - 4684 * b;
        return VP8ClipUV(v, rounding);
}

static __global__ void kernel_rgb_to_yuv(uint8_t *d_out, const uint8_t *d_in,
                                         int width, int height)
{
        size_t x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
        size_t y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
        if (x >= width || y >= height) {
                return;
        }
        int rr = 0;
        int gg = 0;
        int bb = 0;
        for (int j = 0; j < 2; ++j) {
                for (int i = 0; i < 2; ++i) {
                        size_t position = MIN(x + j, (size_t)width - 1) +
                                          (MIN(y + i, (size_t)height - 1) *
                                           width);
                        int r = d_in[3 * position];
                        int g = d_in[(3 * position) + 1];
                        int b = d_in[(3 * position) + 2];
                        const int luma = VP8RGBToY(r, g, b, YUV_HALF);
                        d_out[position] = luma;
                        rr += r;
                        gg += g;
                        bb += b;
                }
        }
        // VP8RGBToU/V expects four accumulated pixels.
        const int u = VP8RGBToU(rr, gg, bb, YUV_HALF << 2);
        const int v = VP8RGBToV(rr, gg, bb, YUV_HALF << 2);
        d_out += (size_t) width * height;
        size_t uv_off = ((y / 2) * ((width + 1) / 2)) + (x / 2);
        d_out[uv_off] = u;
        d_out += (((size_t) width + 1) / 2) * ((height + 1) / 2);
        d_out[uv_off] = v;
}

uint8_t *convert_rgb_to_yuv420(const struct dec_image *in, cudaStream_t stream)
{
        assert(in->comp_count == 3);
        dim3 block(16, 16);
        int width = (in->width + 1) / 2;
        int height = (in->height + 1) / 2;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        size_t len = ((size_t) in->width * in->height) +
                     (2ULL * ((in->width + 1) / 2) * ((in->height + 1) / 2));
        if (len > state.d_yuv420_allocated) {
                CHECK_CUDA(cudaFree(state.d_yuv420));
                CHECK_CUDA(cudaMalloc(&state.d_yuv420, len));
                state.d_yuv420_allocated = len;
        }
        kernel_rgb_to_yuv<<<grid, block, 0, stream>>>(
            state.d_yuv420, in->data, in->width, in->height);
        CHECK_CUDA(cudaGetLastError());
        return state.d_yuv420;
}

static __global__ void kernel_y(const uint8_t *d_in, uint8_t *d_out, size_t count)
{
        int pos = blockIdx.x * blockDim.x + threadIdx.x;
        if (pos >= count) {
                return;
        }
        int val  = d_in[pos];
        d_out[pos] = VP8RGBToY(val, val, val, YUV_HALF);
}
uint8_t *convert_y_full_to_limited(const struct dec_image *in,
                                   cudaStream_t stream)
{
        assert(in->comp_count == 1);
        dim3 block(256, 1);
        size_t count = (size_t) in->width * in->height;
        if (count > state.d_yuv420_allocated) {
                CHECK_CUDA(cudaFree(state.d_yuv420));
                CHECK_CUDA(cudaMalloc(&state.d_yuv420, count));
                state.d_yuv420_allocated = count;
        }
        dim3 grid((count + block.x - 1) / block.x, 1);
        kernel_y<<<grid, block, 0, stream>>>(in->data, state.d_yuv420, count);
        CHECK_CUDA(cudaGetLastError());
        return state.d_yuv420;
}

static __global__ void fillQuadKernel(unsigned char *out, int w, int h,
                                      float2 A, float2 B, float2 C, float2 D)
{
        struct fns {
                static __device__ bool pointInTriangle(const float2 &P,
                                                       const float2 &A,
                                                       const float2 &B,
                                                       const float2 &C)
                {
                        // Barycentric method or cross product winding
                        float2 v0 = make_float2(B.x - A.x, B.y - A.y);
                        float2 v1 = make_float2(C.x - A.x, C.y - A.y);
                        float2 v2 = make_float2(P.x - A.x, P.y - A.y);
                        float d00 = v0.x * v0.x + v0.y * v0.y;
                        float d01 = v0.x * v1.x + v0.y * v1.y;
                        float d11 = v1.x * v1.x + v1.y * v1.y;
                        float d20 = v2.x * v0.x + v2.y * v0.y;
                        float d21 = v2.x * v1.x + v2.y * v1.y;
                        float denom = d00 * d11 - d01 * d01;
                        float v = (d11 * d20 - d01 * d21) / denom;
                        float w = (d00 * d21 - d01 * d20) / denom;
                        float u = 1.0f - v - w;
                        return (u >= 0) && (v >= 0) && (w >= 0);
                }

                static __device__ bool
                pointInQuad(const float2 &P, const float2 &A, const float2 &B,
                            const float2 &C, const float2 &D)
                {
                        // Assumes quad is convex: AB, BC, CD, DA
                        // Test triangles: ABC and CDA
                        return pointInTriangle(P, A, B, C) ||
                               pointInTriangle(P, C, D, A);
                }
        };

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= w || y >= h) {
                return;
        }

        float2 P = make_float2(x + 0.5f, y + 0.5f);
        if (fns::pointInQuad(P, A, B, C, D)) {
                out[y * w + x] = 255;
        } else {
                out[y * w + x] = 0;
        }
}

void rotate_set_alpha(struct dec_image *in, double aDstQuad[4][2],
                                   cudaStream_t stream) {
        float2 quadA = make_float2(aDstQuad[0][0], aDstQuad[0][1]);
        float2 quadB = make_float2(aDstQuad[1][0], aDstQuad[1][1]);
        float2 quadC = make_float2(aDstQuad[2][0], aDstQuad[2][1]);
        float2 quadD = make_float2(aDstQuad[3][0], aDstQuad[3][1]);

        dim3 block(16, 16);
        dim3 grid((in->width + block.x - 1) / block.x, (in->height + block.y - 1) / block.y);
        fillQuadKernel<<<grid, block, 0, stream>>>(
            in->alpha, in->width, in->height, quadA, quadB, quadC, quadD);
        CHECK_CUDA(cudaGetLastError());
}

void cleanup_cuda_kernels()
{
        for (unsigned i = 0; i < ARR_SIZE(state.stat); ++i) {
                CHECK_CUDA(cudaFreeHost(state.stat[i].data));
                CHECK_CUDA(cudaFree(state.stat[i].d_res));
        }
        CHECK_CUDA(cudaFree(state.d_yuv420));
}

struct process_s2 {
        __device__ void operator()(uint16_t &x) const
        {
                x = __saturatef((x + BOA_ADD_OFFSET) /
                                BOA_QUANTIFICATION_VALUE) *
                    65535.0;
        }
};

void thrust_process_s2(uint16_t *d_ptr, size_t count, cudaStream_t stream)
{
        thrust::for_each(thrust::cuda::par.on(stream), d_ptr, d_ptr + count,
                         process_s2{});
}

struct convert_16b_to_8b {
        __device__ uint8_t operator()(const uint16_t &x) const
        {
                return x >> 8;
        }
};

void thrust_16b_to_8b(uint16_t *d_in, uint8_t *d_out, size_t count,
                      cudaStream_t stream) {
        thrust::transform(thrust::cuda::par.on(stream), d_in, d_in + count, d_out,
                         convert_16b_to_8b{});
}
