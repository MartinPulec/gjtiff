#include "nd_processing.cuh"

#include <cassert>
#include <cmath>

#include "cuda_common.cuh"    // for bilinear_sample
#include "defs.h"
#include "utils.h"

struct ramp_item {
        float val;
        int col;
};

static const __constant__ struct ramp_item ramp_ndvi[] = {
        {-0.5, 0x0c0c0c},
        {-0.2, 0xbfbfbf},
        {-0.1, 0xdbdbdb},
        {0, 0xeaeaea},
        {0.025, 0xfff9cc},
        {0.05, 0xede8b5},
        {0.075, 0xddd89b},
        {0.1, 0xccc682},
        {0.125, 0xbcb76b},
        {0.15, 0xafc160},
        {0.175, 0xa3cc59},
        {0.2, 0x91bf51},
        {0.25, 0x7fb247},
        {0.3, 0x70a33f},
        {0.35, 0x609635},
        {0.4, 0x4f892d},
        {0.45, 0x3f7c23},
        {0.5, 0x306d1c},
        {0.55, 0x216011},
        {0.6, 0x0f540a},
        {1, 0x004400},
        {INFINITY, 0x004400},
};

static const __constant__ struct ramp_item ramp_ndmi[] = {
       {-0.8, 0x800000},
       {-0.24, 0xff0000},
       {-0.032, 0xffff00},
       {0.032, 0x00ffff},
       {0.24, 0x0000ff},
       {0.8, 0x000080},
       {INFINITY, 0x000080},
};

static const __constant__ struct ramp_item ramp_ndwi[] = {
       {-0.8, 0x008000},
       {0, 0xFFFFFF},
       {0.8, 0x0000CC},
       {INFINITY, 0x0000CC},
};

template <enum nd_feature feature>
__device__ void process_mapping(uint8_t *out, float val);

template <> __device__ void process_mapping<ND_UNKNOWN>(uint8_t *out, float val)
{
        val = __saturatef((val + 1.F) / 2.F);
#ifdef GAMMA
        val = pow(val, GAMMA);
#endif
        uint8_t grayscale = 255 * val;
        out[0] = grayscale;
        out[1] = grayscale;
        out[2] = grayscale;
}

__device__ static void process_mapping_nd(uint8_t *out, float val,
                                          const struct ramp_item *ramp)
{
        int col = ramp[0].col;
        while (ramp->val != INFINITY) {
                if (val > ramp->val && val <= ramp[1].val) {
                        col = ramp->col;
                        break;
                }
                ramp++;
        }
        uint8_t r = col >> 16;
        uint8_t g = (col >> 8) & 0xff;
        uint8_t b = col & 0xff;
#ifdef GAMMA
        // r = 255.0 * pow(r / 255.0, GAMMA);
        // g = 255.0 * pow(g / 255.0, GAMMA);
        // b = 255.0 * pow(b / 255.0, GAMMA);
#endif
        out[0] = r;
        out[1] = g;
        out[2] = b;
}

__device__ static void
process_mapping_nd_interpolate(uint8_t *out, float val,
                               const struct ramp_item *ramp)
{
        int col = 0;
        // val -= 0.1;
        if (val < ramp[0].val) {
                col = ramp[0].col;
        } else {
                while (ramp->val != INFINITY) {
                        if (val > ramp->val && val <= ramp[1].val) {
                                break;
                        }
                        ramp++;
                }
                if (ramp->val == INFINITY) {
                        col = ramp->col;
                } else {
                        col = ramp->col;
                        uint8_t r1 = col >> 16;
                        uint8_t g1 = (col >> 8) & 0xff;
                        uint8_t b1 = col & 0xff;
                        col = ramp[1].col;
                        uint8_t r2 = col >> 16;
                        uint8_t g2 = (col >> 8) & 0xff;
                        uint8_t b2 = col & 0xff;
                        float scale =  ramp[1].val - ramp->val;
                        val -= ramp->val;
                        uint8_t r = round(r1 * (scale - val) / scale + r2 * val / scale);
                        uint8_t g = round(g1 * (scale - val) / scale + g2 * val / scale);
                        uint8_t b = round(b1 * (scale - val) / scale + b2 * val / scale);
                        // if (ramp->val == 0) printf("%d %d = %d  %d %d   %f %f   %f\n", b1, b2, r,g,b,  ramp[0].val, ramp[1].val, val + ramp->val);
                        col = r << 16 | g << 8 | b;
                }
        }
        uint8_t r = col >> 16;
        uint8_t g = (col >> 8) & 0xff;
        uint8_t b = col & 0xff;
#ifdef GAMMA
        // r = 255.0 * pow(r / 255.0, GAMMA);
        // g = 255.0 * pow(g / 255.0, GAMMA);
        // b = 255.0 * pow(b / 255.0, GAMMA);
#endif
        out[0] = r;
        out[1] = g;
        out[2] = b;
}
template <> __device__ void process_mapping<NDVI>(uint8_t *out, float val)
{
        process_mapping_nd(out, val, ramp_ndvi);
}

template <> __device__ void process_mapping<NDMI>(uint8_t *out, float val)
{
        process_mapping_nd_interpolate(out, val, ramp_ndmi);
}


template <> __device__ void process_mapping<NDWI>(uint8_t *out, float val)
{
        process_mapping_nd_interpolate(out, val, ramp_ndwi);
}

template <enum nd_feature feature>
static __global__ void nd_process(struct dec_image out, struct dec_image in1,
                                  struct dec_image in2, int d_fill_color)
{
        int out_x = blockIdx.x * blockDim.x + threadIdx.x; // column index
        int out_y = blockIdx.y * blockDim.y + threadIdx.y; // row index

        if (out_x >= out.width || out_y >= out.height) {
                return;
        }
        uint8_t *out_p = out.data + ((size_t)3 * (out_x + (out_y * out.width)));

        float rel_pos_src_x = (out_x + 0.5) / out.width;
        float rel_pos_src_y = (out_y + 0.5) / out.height;

        float abs_pos_src_x = rel_pos_src_x * in1.width;
        float abs_pos_src_y = rel_pos_src_y * in1.height;
        uint8_t alpha1 = bilinearSample(in1.alpha, in1.width, 1, in1.height,
                                        abs_pos_src_x, abs_pos_src_y);
        out.alpha[out_x + (out_y * out.width)] =  alpha1;
        if (alpha1 != 255) {
                out_p[0] = out_p[1] = out_p[2] = d_fill_color;
                return;
        }

        float val1 = bilinearSample(
            in1.data, in1.width, 1, in1.height, abs_pos_src_x, abs_pos_src_y);

        abs_pos_src_x = rel_pos_src_x * in2.width;
        abs_pos_src_y = rel_pos_src_y * in2.height;
        float val2 = bilinearSample(
            in2.data, in2.width, 1, in2.height, abs_pos_src_x, abs_pos_src_y);

        val1 /= 255.0F;
        val2 /= 255.0F;
#ifdef GAMMA
        val1 = pow(val1, 1.0F / GAMMA);
        val2 = pow(val2, 1.0F / GAMMA);
#endif
        val1 *= in1.scale;
        val2 *= in2.scale;

        float res = (val1 - val2) / (val1 + val2 + 0.000000001f);

        process_mapping<feature>(out_p, res);
}

void process_nd_features_cuda(struct dec_image *out, enum nd_feature feature,
                              const struct dec_image *in1,
                              const struct dec_image *in2, cudaStream_t stream)
{
        assert(alpha_wanted);
        GPU_TIMER_START(process_nd_features_cuda, LL_DEBUG, stream);
        dim3 block(16, 16);
        int width = out->width;
        int height = out->height;
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
        auto *fn = nd_process<ND_UNKNOWN>;
        switch (feature) {
        case NDVI:
                fn =  nd_process<NDVI>;
                break;
        case NDMI:
                fn = nd_process<NDMI>;
                break;
        case NDWI:
                fn = nd_process<NDWI>;
                break;
        case ND_UNKNOWN: // already set
                break;
        }
        fn<<<grid, block, 0, stream>>>(*out, *in1, *in2, fill_color);
        CHECK_CUDA(cudaGetLastError());
        GPU_TIMER_STOP(process_nd_features_cuda);
}
